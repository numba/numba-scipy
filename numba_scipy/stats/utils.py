from numba.targets.registry import cpu_target
from numba.extending import typeof_impl
from numba import types

def overload_pyclass(pyclass: type, jitclass: type):
    """
    Tells Numba to use jitclass instead of pyclass while in jitted in code

    :param pyclass: python class to be overloaded
    :param jitclass: jitclass Numba should use instead of pyclass
    :return: None
    """
    assert hasattr(jitclass, "class_type"), (
        "jitclass argument must be a Numba jitclass (not an instance)")
    _overload_pyclass1(pyclass, jitclass)
    _overload_pyclass2(pyclass, jitclass)
    unbox_pyclass(pyclass, jitclass)


def _overload_pyclass1(pyclass: type, jitclass: type):
    """
    Tells Numba to infer some Python values as instances of that type
    """
    @typeof_impl.register(pyclass)
    def typeof_index(val, c):
        return jitclass.class_type.instance_type


def _overload_pyclass2(pyclass: type, jitclass: type):
    """
    Register resolution of the class object
    """
    typingctx = cpu_target.typing_context
    typingctx.insert_global(pyclass, jitclass.class_type)


from numba.jitclass.base import imp_dtor


def unbox_pyclass(pyclass: type, jitclass: type):
    """
    Register custom unbox procedure

    """
    from numba.extending import unbox, NativeValue
    from numba import cgutils

    # undo automatic registration of unbox function
    from numba.pythonapi import _unboxers
    del _unboxers.functions[types.ClassInstanceType]

    @unbox(types.ClassInstanceType)
    def unbox_interval(typ, obj, c):
        """
        Convert a Interval object to a native interval structure.
        """
        obj_list = []
        type_inst_list = []
        for attr_name, attr_typ in typ.struct.items():
            obj_list.append(c.pyapi.object_getattr_string(obj, attr_name))
            type_inst_list.append(attr_typ)

        type_inst_list = tuple(type_inst_list)

        # Allocate the instance
        inst_typ = typ
        context = c.context
        builder = c.builder
        alloc_type = context.get_data_type(inst_typ.get_data_type())
        alloc_size = context.get_abi_sizeof(alloc_type)

        meminfo = context.nrt.meminfo_alloc_dtor(builder, context.get_constant(types.uintp, alloc_size),
            imp_dtor(context, builder.module, inst_typ), )
        data_pointer = context.nrt.meminfo_data(builder, meminfo)
        data_pointer = builder.bitcast(data_pointer, alloc_type.as_pointer())

        # Nullify all data
        builder.store(cgutils.get_null_value(alloc_type), data_pointer)

        inst_struct = context.make_helper(builder, inst_typ)
        inst_struct.meminfo = meminfo
        inst_struct.data = data_pointer

        # Prepare return value
        ret = inst_struct._getvalue()

        return NativeValue(ret, is_error=c.pyapi.c_api_error())


__all__ = [overload_pyclass]
