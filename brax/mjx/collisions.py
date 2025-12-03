import copy
from typing import Optional, Union

import jax
import mujoco
import numpy as np
from jax import numpy as jp
from mujoco.mjx._src import collision_driver
from mujoco.mjx._src import math
from mujoco.mjx._src import mesh
from mujoco.mjx._src import types
from mujoco.mjx._src.collision_driver import _COLLISION_FUNC
from mujoco.mjx._src.collision_sdf import _optim, _cylinder, collider
from mujoco.mjx._src.collision_types import Collision
from mujoco.mjx._src.collision_types import GeomInfo
from mujoco.mjx._src.io import _resolve_impl_and_device, _put_option, _put_statistic, _strip_weak_type
from mujoco.mjx._src.types import GeomType


def put_model(
        m: mujoco.MjModel,
        device: Optional[jax.Device] = None,
        impl: Optional[Union[str, types.Impl]] = None,
        _full_compat: bool = False,  # pylint: disable=invalid-name
) -> types.Model:
    """Puts mujoco.MjModel onto a device, resulting in mjx.Model.

    Args:
      m: the model to put onto device
      device: which device to use - if unspecified picks the default device
      impl: implementation to use
      _full_compat: put all MjModel fields onto device irrespective of MJX support
        This is an experimental feature.  Avoid using it for now.

    Returns:
      an mjx.Model placed on device

    Raises:
      ValueError: if impl is not supported
      DeprecationWarning: if _full_compat is True
    """

    impl, device = _resolve_impl_and_device(impl, device)
    return put_model_jax(m, device)


def box(pos: jax.Array, size: jax.Array) -> jax.Array:
    """Signed distance to an oriented box with half-extents = size."""
    q = jp.abs(pos) - size  # size = [hx, hy, hz]
    outside = jp.maximum(q, 0.0)
    inside = jp.minimum(jp.maximum(q[0], jp.maximum(q[1], q[2])), 0.0)
    return math.norm(outside) + inside


@collider(ncon=1)
def cylinder_box(c: GeomInfo, b: GeomInfo) -> Collision:
    """Cylinder vs Box via SDF clearance optimization."""
    x0 = 0.5 * (c.pos + b.pos)
    return _optim(_cylinder, box, c, b, x0)


@collider(ncon=1)
def box_cylinder(b: GeomInfo, c: GeomInfo) -> Collision:
    """Box vs Cylinder; just flip args into the same optimizer."""
    x0 = 0.5 * (b.pos + c.pos)
    return _optim(box, _cylinder, b, c, x0)


_COLLISION_FUNC[(GeomType.BOX, GeomType.CYLINDER)] = box_cylinder
_COLLISION_FUNC[(GeomType.CYLINDER, GeomType.BOX)] = cylinder_box


def has_collision_fn(t1: GeomType, t2: GeomType) -> bool:
    """Returns True if a collision function exists for a pair of geom types."""
    return (t1, t2) in _COLLISION_FUNC


def put_model_jax(
        m: mujoco.MjModel,
        device: Optional[jax.Device] = None,
) -> types.Model:
    """Puts mujoco.MjModel onto a device, resulting in mjx.Model."""
    if m.nflex:
        raise NotImplementedError('Flex not implemented for JAX backend.')

    # contact sensor
    is_contact_sensor = m.sensor_type == types.SensorType.CONTACT
    if is_contact_sensor.any():
        objtype = m.sensor_objtype[is_contact_sensor]
        reftype = m.sensor_reftype[is_contact_sensor]
        contact_sensor_type = set(np.concatenate([objtype, reftype]))

        # site filter
        if types.ObjType.SITE in set(objtype):
            raise NotImplementedError(
                'Contact sensor with site matching semantics not implemented for JAX'
                ' backend.'
            )

        # body semantics
        if types.ObjType.BODY in contact_sensor_type:
            raise NotImplementedError(
                'Contact sensor with body matching semantics not implemented for JAX'
                ' backend.'
            )

        # subtree semantics
        if types.ObjType.XBODY in contact_sensor_type:
            raise NotImplementedError(
                'Contact sensor with subtree matching semantics not implemented for'
                ' JAX backend.'
            )

        # net force
        if (m.sensor_intprm[is_contact_sensor, 1] == 3).any():
            raise NotImplementedError(
                'Contact sensor with netforce reduction not implemented for JAX'
                ' backend.'
            )

    mesh_geomid = set()
    for g1, g2, ip in collision_driver.geom_pairs(m):
        t1, t2 = m.geom_type[[g1, g2]]
        # check collision function exists for type pair
        if not has_collision_fn(t1, t2):
            t1, t2 = mujoco.mjtGeom(t1), mujoco.mjtGeom(t2)
            raise NotImplementedError(f'({t1}, {t2}) collisions not implemented.')
        # margin/gap not supported for meshes and height fields
        no_margin = {mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_HFIELD}
        if no_margin.intersection({t1, t2}):
            if ip != -1:
                margin = m.pair_margin[ip]
            else:
                margin = m.geom_margin[g1] + m.geom_margin[g2]
            if margin.any():
                t1, t2 = mujoco.mjtGeom(t1), mujoco.mjtGeom(t2)
                raise NotImplementedError(f'({t1}, {t2}) margin/gap not implemented.')
        for t, g in [(t1, g1), (t2, g2)]:
            if t == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_geomid.add(g)

    for enum_field, enum_type, mj_type in (
            (m.actuator_biastype, types.BiasType, mujoco.mjtBias),
            (m.actuator_dyntype, types.DynType, mujoco.mjtDyn),
            (m.actuator_gaintype, types.GainType, mujoco.mjtGain),
            (m.actuator_trntype, types.TrnType, mujoco.mjtTrn),
            (m.eq_type, types.EqType, mujoco.mjtEq),
            (m.sensor_type, types.SensorType, mujoco.mjtSensor),
            (m.wrap_type, types.WrapType, mujoco.mjtWrap),
    ):
        missing = set(enum_field) - set(enum_type)
        if missing:
            raise NotImplementedError(
                f'{[mj_type(m) for m in missing]} not supported'
            )

    mj_field_names = {f.name for f in types.Model.fields() if f.name != '_impl'}
    fields = {f: getattr(m, f) for f in mj_field_names}
    fields['cam_mat0'] = fields['cam_mat0'].reshape((-1, 3, 3))
    fields['opt'] = _put_option(m.opt, types.Impl.JAX)
    fields['stat'] = _put_statistic(m.stat, types.Impl.JAX)

    fields_jax = {}
    fields_jax['dof_hasfrictionloss'] = fields['dof_frictionloss'] > 0
    fields_jax['tendon_hasfrictionloss'] = fields['tendon_frictionloss'] > 0
    fields_jax['geom_rbound_hfield'] = fields['geom_rbound']

    # spatial tendon wrap inside
    fields_jax['wrap_inside_maxiter'] = 5
    fields_jax['wrap_inside_tolerance'] = 1.0e-4
    fields_jax['wrap_inside_z_init'] = 1.0 - 1.0e-5
    fields_jax['is_wrap_inside'] = np.zeros(0, dtype=bool)
    if m.nsite:
        # find sphere or cylinder geoms (if any exist)
        (wrap_id_geom,) = np.nonzero(
            (m.wrap_type == mujoco.mjtWrap.mjWRAP_SPHERE)
            | (m.wrap_type == mujoco.mjtWrap.mjWRAP_CYLINDER)
        )
        wrap_objid_geom = m.wrap_objid[wrap_id_geom]
        geom_pos = m.geom_pos[wrap_objid_geom]
        geom_size = m.geom_size[wrap_objid_geom, 0]

        # find sidesites (if any exist)
        side_id = np.round(m.wrap_prm[wrap_id_geom]).astype(int)
        side = m.site_pos[side_id]

        # wrap inside flag
        fields_jax['is_wrap_inside'] = np.array(
            (np.linalg.norm(side - geom_pos, axis=1) < geom_size) & (side_id >= 0)
        )

    # Pre-compile meshes for MJX collisions.
    fields_jax['mesh_convex'] = [None] * m.nmesh
    for i in mesh_geomid:
        dataid = m.geom_dataid[i]
        if fields_jax['mesh_convex'][dataid] is None:
            fields_jax['mesh_convex'][dataid] = mesh.convex(m, dataid)  # pytype: disable=unsupported-operands
    fields_jax['mesh_convex'] = tuple(fields_jax['mesh_convex'])

    jax_impl = types.ModelJAX(**fields_jax)
    model = types.Model(
        **{k: copy.copy(v) for k, v in fields.items()}, _impl=jax_impl
    )

    model = jax.device_put(model, device=device)
    return _strip_weak_type(model)
