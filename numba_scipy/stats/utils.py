from numba.targets.registry import cpu_target
from numba.extending import typeof_impl
from numba.jitclass.base import imp_dtor
from numba.targets.imputils import lower_constant
from numba import cgutils, types


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
    _lower_constant_jitclass(jitclass)
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


def _mangle_attr(name):
    """
    Mangle attributes.
    The resulting name does not startswith an underscore '_'.
    """
    return 'm_' + name


def _lower_constant_jitclass(jitclass: type):

    @lower_constant(jitclass.class_type.instance_type)
    def _lower_constant_class_instance(context, builder, typ, pyval):
        # Allocate the instance
        inst_typ = typ

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

        # Assign value
        for attr_name in typ.struct:
            data = context.make_data_helper(builder, typ.get_data_type(),
                                            ref=data_pointer)
            attr_type = typ.struct[attr_name]
            attr_pyval = getattr(pyval, attr_name)
            val = context.get_constant_generic(builder, attr_type, attr_pyval)
            setattr(data, _mangle_attr(attr_name), val)

        # Prepare return value
        ret = inst_struct._getvalue()

        context.nrt.incref(builder, typ, ret)
        return ret


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
    def unbox_impl(typ, obj, c):
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

        # Assign value
        for attr_name in typ.struct:
            data = context.make_data_helper(builder, typ.get_data_type(),
                                            ref=data_pointer)
            attr_type = typ.struct[attr_name]
            generic_type = type(attr_type)
            unbox_fc = _unboxers.functions[generic_type]
            attr_obj = c.pyapi.object_getattr_string(obj, attr_name)
            native_val = unbox_fc(attr_type, attr_obj, c)
            setattr(data, _mangle_attr(attr_name), native_val.value)

        # Prepare return value
        ret = inst_struct._getvalue()

        return NativeValue(ret, is_error=c.pyapi.c_api_error())


__all__ = [overload_pyclass]
