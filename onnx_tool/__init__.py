import os
import warnings

import numpy
import onnx

from .graph import Graph
from .node_profilers import NodeBase, node_profile, node_infer_shape, Constant
from .tensors import graph_addoutputs, graph_set_inputs, shape_of_tensor, is_valid_ndarray, tensorproto2ndarray, volume, \
    create_ndarray_f32, create_ndarray_int64, update_static_tensors
from .utils import NODEPROFILER_REGISTRY, timer, tuple2str, GLOBAL_VARS, VERSION


def __remove_initilisers(model: onnx.ModelProto):
    model.graph.ClearField('initializer')


def __remove_constantnodes(model: onnx.ModelProto):
    validnodes = []
    for node in model.graph.node:
        if node.op_type != 'Constant':
            validnodes.append(node)
    model.graph.ClearField('node')
    for node in validnodes:
        model.graph.node.append(node)


def model_export_tensors_numpy(m, tensornames: [str] = None, savefolder: str = None, fp16: bool = False) -> None:
    if isinstance(m, str):
        m = onnx.load_model(m)
    GLOBAL_VARS['tensor_map'] = {}
    GLOBAL_VARS['params_map'] = {}

    def save_numpy(arr: numpy.ndarray, fp16: bool, filename):
        if fp16 and arr.dtype in [numpy.float32, numpy.float64]:
            arr = arr.astype(numpy.float16)
        numpy.save(filename, arr)

    if isinstance(m, onnx.ModelProto):
        update_static_tensors(m.graph)
        if savefolder is not None:
            os.makedirs(savefolder, exist_ok=True)
        else:
            savefolder = './'
        tensor_map = GLOBAL_VARS['tensor_map']
        if tensornames is None:
            for key in tensor_map.keys():
                name = key
                if '/' in key:
                    name = key.replace('/', '_')
                if '\\' in key:
                    name = key.replace('\\', '_')
                save_numpy(tensor_map[key], fp16, os.path.join(savefolder, name + '.npy'))

        else:
            for name in tensornames:
                if name not in tensor_map.keys():
                    warnings.warn(f'tensor {name} not found ')
                    continue
                fname = name
                if '/' in name:
                    fname = name.replace('/', '_')
                if '\\' in name:
                    fname = name.replace('\\', '_')
                save_numpy(tensor_map[name], fp16, os.path.join(savefolder, fname + '.npy'))


def infer_shapes(graph: onnx.GraphProto, dynamic_tensors: {}, verbose: bool = False) -> [map, map]:
    """
        Returns: {TensorName:ndarray},{NodeName:int}
    """
    GLOBAL_VARS['tensor_map'] = {}
    GLOBAL_VARS['params_map'] = {}

    update_static_tensors(graph)

    if dynamic_tensors is not None:
        graph_set_inputs(graph, dynamic_tensors)

    tensor_map = GLOBAL_VARS['tensor_map']
    params_map = GLOBAL_VARS['params_map']
    for input in graph.input:
        shape = shape_of_tensor(input)
        for d in shape:
            if d < 0:
                raise ValueError(f"Input {input.name}'s shape is dynamic, please set it a fixed input dimension")
        if input.name not in tensor_map:
            tensor_map.update({input.name: numpy.zeros(shape, dtype=numpy.float32)})

        if not is_valid_ndarray(tensor_map[input.name]):
            raise ValueError(f"Input {input.name}'s shape is dynamic, please set it a fixed input dimension")

    # itmr = timer()
    # for initial in graph.initializer:
    #     arr=tensorproto2ndarray(initial)
    #     tensor_map.update({initial.name:arr})
    #     vol=volume(arr.shape)
    #     if vol==0:#scalar
    #         vol=1
    #     params_map.update({initial.name:vol})
    #     if verbose:
    #         print(initial.name, itmr.stop(), arr.shape)

    for node in graph.node:
        ins = []
        for input in node.input:
            if input == '':
                continue
            ins.append(tensor_map[input])
        outs = []
        for output in node.output:
            if output == '':
                continue
            outs.append(output)
        outtensors = node_infer_shape(node, ins)
        for tensor, name in zip(outtensors, outs):
            tensor_map[name] = tensor

    for key in tensor_map.keys():
        shape = tensor_map[key].shape
        if len(shape) == 0:
            shape = (0,)
        vinf = onnx.helper.make_tensor_value_info(key, onnx.TensorProto.FLOAT, shape)
        graph.value_info.append(vinf)

    for output in graph.output:
        dim = output.type.tensor_type.shape.dim
        for nb, dnb in zip(dim, tensor_map[output.name].shape):
            nb.dim_value = dnb
    GLOBAL_VARS['tensor_map'] = tensor_map
    GLOBAL_VARS['params_map'] = params_map
    return tensor_map, params_map


def graph_profile(graph: onnx.GraphProto, dynamic_shapes: {}, verbose=False, hidden_ops: [str] = None,
                  ) -> [float, float, map]:
    """
        return MACs,Params,NodeMap
    """
    macs = 0.0
    params = 0
    memory = 0

    gtmr = timer()

    gtmr.start()
    tmap, pmap = infer_shapes(graph, dynamic_shapes, verbose=verbose)
    if verbose:
        print(f'infered all tensor shapes, time cost {gtmr.stop():.3f} s')

    node_map = {}
    index = 0
    gtmr.start()
    params_map = GLOBAL_VARS['params_map']
    params_flag_map = {}
    for key in params_map.keys():
        params_flag_map[key] = 0
    params_shared_nodes = {}
    for input in graph.input:
        tensor = tmap[input.name]
        _memory = volume(tensor.shape) * 4
        node_map.update({input.name: {'macs': 0, 'params': 0, 'memory': _memory, 'inshape': tensor.shape,
                                      'outshape': tensor.shape}})
        memory += _memory

    for node in graph.node:
        ins = []
        _params = 0
        _memory = 0
        if hidden_ops is not None:
            if node.op_type in hidden_ops:
                continue
        for input in node.input:
            if input == '':
                continue
            ins.append(tmap[input])
            if input in pmap.keys():
                if params_flag_map[input] == 0:
                    _params += pmap[input]
                    _memory += pmap[input]

                params_flag_map[input] += 1

        outs = []
        for output in node.output:
            if tmap.keys().__contains__(output):
                outs.append(tmap[output])
                if node.op_type == 'Constant':
                    # Constant's output tensors are already counted as weight tensors
                    continue
                _memory += volume(tmap[output].shape)
        _macs, _params_c = node_profile(node, ins, outs)
        # @deprecated _params_c

        outshape = (0,)
        if len(outs) > 0:
            outshape = outs[0].shape
            outshape = (0,) if len(outshape) == 0 else outshape
        inshape = (0,)
        if len(ins) > 0:
            inshape = ins[0].shape
            inshape = (0,) if len(inshape) == 0 else inshape
        if len(node.name) == 0:
            node.name = node.op_type + '_{}'.format(index)
        index += 1
        _memory *= 4
        node_map.update({node.name: {'macs': _macs, 'params': _params, 'memory': _memory, 'inshape': inshape,
                                     'outshape': outshape}})
        macs += _macs
        params += _params
        memory += _memory
    if verbose:
        print(f'profile all nodes, time cost {gtmr.stop():.3f} s')

    for node in graph.node:
        for input in node.input:
            if input == '':
                continue
            if input in pmap.keys():
                if params_flag_map[input] > 1 and volume(tmap[input].shape) > 0:
                    if input in params_shared_nodes:
                        params_shared_nodes[input].append(node.name)
                    else:
                        params_shared_nodes[input] = [node.name]

    GLOBAL_VARS['macs'] = macs
    GLOBAL_VARS['params'] = params
    GLOBAL_VARS['memory'] = memory
    GLOBAL_VARS['node_map'] = node_map
    GLOBAL_VARS['params_shared_nodes'] = params_shared_nodes
    if verbose:
        tmem_count = 0
        for t in tmap:
            tmem_count += volume(tmap[t].shape)
        tmem_count *= 4
        diffratio = abs(memory - tmem_count) / tmem_count
        print(
            f'Memory sum from TensorMap:{tmem_count} Memory sum from NodeMap sum:{memory}, diff ratio:{diffratio:.3%}')
        assert (diffratio < 0.01)

    return macs, params, node_map


FusedOps = (
    'Identity', 'Relu', 'LeakyRelu', 'Constant', 'Shape', 'Squeeze', 'Unsqueeze', 'Reshape', 'ConstantOfShape', 'Cast'
)


def model_profile(m, dynamic_shapes: {str: tuple} = None, savenode: str = None,
                  saveshapesmodel: str = None, shapesonly: bool = False, verbose: bool = False,
                  hidden_ops: [str] = (),
                  dump_outputs: [str] = None, remove_unused_tensors=True) -> None:
    if isinstance(m, str):
        m = onnx.load_model(m)
    if isinstance(m, onnx.ModelProto):
        if remove_unused_tensors:
            graph_remove_unused_tensors(m.graph)
        graph_profile(m.graph, dynamic_shapes, verbose, hidden_ops=hidden_ops)
        print_node_map(savenode)
        if saveshapesmodel is not None:
            if shapesonly:
                __remove_initilisers(m)
                __remove_constantnodes(m)

            if dump_outputs is not None:
                graph_addoutputs(m.graph, dump_outputs)
            G = Graph(m.graph)
            G.save_model(saveshapesmodel)


def model_shape_infer(m, dynamic_shapes: {str: tuple} = None,
                      saveshapesmodel: str = None, shapesonly: bool = False, verbose: bool = False,
                      dump_outputs: [str] = None):
    if isinstance(m, str):
        m = onnx.load_model(m)
    if isinstance(m, onnx.ModelProto):
        _, _ = infer_shapes(m.graph, dynamic_shapes, verbose)
        if saveshapesmodel is not None:
            if shapesonly:
                __remove_initilisers(m)
                __remove_constantnodes(m)

            if dump_outputs is not None:
                graph_addoutputs(m.graph, dump_outputs)
            G = Graph(m.graph)
            G.save_model(saveshapesmodel)


def print_node_map(f: str = None, metric='MACs'):
    from tabulate import tabulate
    assert (metric in ['MACs', 'FLOPs'])
    node_map = GLOBAL_VARS['node_map']
    saveformat = 'txt'
    splitch = 'x'

    if f is not None and '.csv' in f:
        saveformat = 'csv'

    ptable = []

    macs = int(round(GLOBAL_VARS['macs']))
    params = int(GLOBAL_VARS['params'])
    memory = int(GLOBAL_VARS['memory'])

    shared_params = GLOBAL_VARS['params_shared_nodes']
    if len(shared_params.keys()):
        print()
        print('*' * 64)
        print(f'Please note that Weight Tensors Sharing is detected:')
        for key in shared_params.keys():
            print(f'Tensor:{key} ')
            print('Shared by: ')
            for node in shared_params[key]:
                print('           ', node)
            print()
        print('*' * 64)

    factor = 1
    if metric == 'FLOPs':
        factor = 2

    params += 1e-18
    macs += 1e-18
    for key in node_map.keys():
        item = node_map[key]
        if saveformat == 'csv':
            row = [key, '{}'.format(
                int(item['macs']) * factor), '{:.2%}'.format(item['macs'] / macs)
                , '{}'.format(int(item['memory'])), '{:.2%}'.format(item['memory'] / memory)
                , '{}'.format(int(item['params'])), '{:.2%}'.format(item['params'] / params),
                   tuple2str(item['inshape'], splitch), tuple2str(item['outshape'], splitch)]
        else:
            row = [key, '{:,}'.format(
                int(item['macs']) * factor), '{:.2%}'.format(item['macs'] / macs)
                , '{:,}'.format(int(item['memory'])), '{:.2%}'.format(item['memory'] / memory)
                , '{:,}'.format(int(item['params'])), '{:.2%}'.format(item['params'] / params),
                   tuple2str(item['inshape'], splitch), tuple2str(item['outshape'], splitch)]
        ptable.append(row)

    if saveformat == 'csv':
        row = ['Total', f'{int(macs * factor)}', '100%', f'{memory}', '100%', f'{int(params)}', '100%', '_', '_']
    else:
        row = ['Total', f'{int(macs * factor):,}', '100%', f'{memory:,}', '100%', f'{int(params):,}', '100%', '_', '_']

    ptable.append(row)
    if f is None:
        print(tabulate(ptable,
                       headers=['Name', metric, 'CPercent', 'Memory', 'MPercent', 'Params', 'PPercent', 'InShape',
                                'OutShape']))
    else:
        fp = open(f, 'w')
        if saveformat == 'csv':
            fp.write(f'Name,{metric},CPercent,Memory,MPercent,Params,PPercent,InShape,OutShape\n')
            for row in ptable:
                str = ''
                for i, ele in enumerate(row):
                    str += ele
                    if i != len(row) - 1:
                        str += ','
                str += '\n'
                fp.write(str)
        else:
            fp.write(tabulate(ptable, headers=['Name', metric, 'CPercent', 'Memory', 'MPercent', 'Params', 'PPercent',
                                               'InShape', 'OutShape']))
        fp.close()


def graph_simplify_names(graph, renametensor=True, renamelayer=True, custom_inputs=None, custom_outputs=None,
                         remove_unused_tensors=True):
    '''
        Args:
            graph: onnx.GraphProto
            renametensor: boolean  eg.: resnetblock1_conv0_weight => 123
            renamelayer: boolean eg.: resnetblock_conv0 => Conv_0
            custom_inputs: [str] | {str:str} eg.: ['input'] without shapes, {'input':'Nx3xwidthxheight'} with shapes
            custom_outputs: [str] | {str:str} eg.: ['output'] without shapes, {'output':'Nx1xwidthxheight'} with shapes
        Returns:

    '''
    if remove_unused_tensors:
        graph_remove_unused_tensors(graph)
    if renamelayer:
        count = 0
        for node in graph.node:
            node.name = node.op_type + '_' + str(count)
            count += 1
    if renametensor:
        total_t = {}
        for node in graph.node:
            for input in node.input:
                total_t[input] = 0
            for output in node.output:
                total_t[output] = 0
        count = 0
        for key in total_t.keys():
            total_t[key] = str(count)
            count += 1

        if custom_inputs is not None:
            if isinstance(custom_inputs, list):
                assert (len(custom_inputs) == len(graph.input))
                for i, input in enumerate(graph.input):
                    total_t[input.name] = custom_inputs[i]
            elif isinstance(custom_inputs, dict):
                keylist = list(custom_inputs.keys())
                assert (len(keylist) == len(graph.input))
                for i, input in enumerate(graph.input):
                    total_t[input.name] = keylist[i]

                    # maybe consider create a new valueinfoproto
                    shapes = custom_inputs[keylist[i]].split('x')
                    dim = input.type.tensor_type.shape.dim
                    assert (len(shapes) == len(dim))
                    for nb, shapeval in zip(dim, shapes):
                        if shapeval.isnumeric():
                            if nb.HasField('dim_param'):
                                nb.ClearField('dim_param')
                            nb.dim_value = int(shapeval)
                        else:
                            if nb.HasField('dim_value'):
                                nb.ClearField('dim_value')
                            nb.dim_param = shapeval
            else:
                raise NotImplementedError()

        if custom_outputs is not None:
            if isinstance(custom_outputs, list):
                assert (len(custom_outputs) == len(graph.output))
                for i, output in enumerate(graph.output):
                    total_t[output.name] = custom_outputs[i]
            elif isinstance(custom_outputs, dict):
                keylist = list(custom_outputs.keys())
                assert (len(keylist) == len(graph.output))
                for i, output in enumerate(graph.output):
                    total_t[output.name] = keylist[i]
                    shapes = custom_outputs[keylist[i]].split('x')
                    # maybe consider create a new valueinfoproto
                    dim = output.type.tensor_type.shape.dim
                    assert (len(shapes) == len(dim))
                    for nb, shapeval in zip(dim, shapes):
                        if shapeval.isnumeric():
                            if nb.HasField('dim_param'):
                                nb.ClearField('dim_param')
                            nb.dim_value = int(shapeval)
                        else:
                            if nb.HasField('dim_value'):
                                nb.ClearField('dim_value')
                            nb.dim_param = shapeval
            else:
                raise NotImplementedError()

        for initial in graph.initializer:
            initial.name = total_t[initial.name]
        for node in graph.node:
            for i, input in enumerate(node.input):
                node.input[i] = total_t[input]
            for i, output in enumerate(node.output):
                node.output[i] = total_t[output]

        for input in graph.input:
            input.name = total_t[input.name]

        for output in graph.output:
            output.name = total_t[output.name]


def graph_remove_unused_tensors(graph):
    producer = {}
    consumer = {}
    for initial in graph.initializer:
        producer[initial.name] = 0
    for node in graph.node:
        for input in node.input:
            consumer[input] = 0
        for output in node.output:
            producer[output] = 0
    inputs = []
    outputs = []
    for key in consumer.keys():
        if key not in producer:
            inputs.append(key)
    for key in producer.keys():
        if key not in consumer:
            outputs.append(key)
    valid_inputs = []
    valid_outputs = []
    for input in graph.input:
        if input.name in inputs:
            valid_inputs.append(input)
    for output in graph.output:
        if output.name in outputs:
            valid_outputs.append(output)
    graph.ClearField('input')
    for input in valid_inputs:
        graph.input.append(input)
    graph.ClearField('output')
    for output in valid_outputs:
        graph.output.append(output)


def model_simplify_names(m, savemodel: str, renametensor=True, renamelayer=True, custom_inputs=None,
                         custom_outputs=None, remove_unused_tensors=True):
    if isinstance(m, str):
        m = onnx.load_model(m)
    if isinstance(m, onnx.ModelProto):
        graph_simplify_names(m.graph, renametensor, renamelayer, custom_inputs, custom_outputs, remove_unused_tensors)
        G = Graph(m.graph)
        G.save_model(savemodel)


def model_subgraph(m, in_tensor_names: [str] = None, out_tensor_names: [str] = None, nodenames: [str] = None,
                   savefolder='./'):
    if isinstance(m, str):
        mname = os.path.basename(m)
        mname = os.path.splitext(mname)[0]
        m = onnx.load_model(m)
    else:
        mname = ''
    if isinstance(m, onnx.ModelProto):
        graph = Graph(m.graph)
        if in_tensor_names is not None and out_tensor_names is not None:
            graph_lvl0, graph_lvl1, graph_lvl2 = graph.get_subgraph(inputs=in_tensor_names, outputs=out_tensor_names)
            graph_lvl0.save_model(os.path.join(savefolder, mname + '_level0.onnx'))
            graph_lvl1.save_model(os.path.join(savefolder, mname + '_level1.onnx'))
            graph_lvl2.save_model(os.path.join(savefolder, mname + '_level2.onnx'))
        if nodenames is not None:
            rawgraph = graph.get_onnxgraph_by_nodenames(nodenames)
            subgraph = Graph(rawgraph)
            subgraph.save_model(os.path.join(savefolder, mname + '_subgraph.onnx'))


def model_opfusion(m, op_type: str, op_name: str, savefile, in_tensor_names: [str] = None,
                   out_tensor_names: [str] = None, nodenames: [str] = None, keep_attr=True):
    if isinstance(m, str):
        m = onnx.load_model(m)

    if isinstance(m, onnx.ModelProto):
        graph = Graph(m.graph)
        if in_tensor_names is not None and out_tensor_names is not None:
            graph = graph.fuse_subgraph_iotensors(inputs=in_tensor_names, outputs=out_tensor_names, name=op_name,
                                                  nodeop=op_type, keep_attr=keep_attr)
            graph.save_model(savefile)
        if nodenames is not None:
            graph = graph.fuse_subgraph_node_names(nodenames, nodeop=op_type, name=op_name, keep_attr=keep_attr)
            graph.save_model(savefile)
