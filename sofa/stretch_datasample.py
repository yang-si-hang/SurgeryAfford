""" 使用sofa采集不同的接触点对软体物体的拉伸数据
created on 2025-11-12
"""
import Sofa
import SofaRuntime
# import Sofa.Gui
# import Sofa.SofaGL
import numpy as np

from deformation_model.diffpd_2d import Soft2D
from utilize.mesh_io import write_mshv2_triangular, write_vtu
from utilize.mesh_util import mesh_obj_tri
from sofa.sofa_utilze import add_move
from const import MESH_DIR, OUTPUT_DIR

def createScene(root, file_path:str, contact:list):
    root.addObject('RequiredPlugin', pluginName=['Sofa.Component',
                                                 'Sofa.Component.Collision',
                                                 'Sofa.Component.Constraint.Projective',
                                                 'Sofa.Component.IO.Mesh',
                                                 'Sofa.Component.LinearSolver',
                                                 'Sofa.GL.Component.Rendering3D'])
    
    root.dt = 0.01
    root.bbox = [[-0.1, -0.1, 0.], [0.2, 0.2, 0.1]]
    root.gravity = [0., 0., 0.]
    root.addObject('VisualStyle', displayFlags='showBehaviorModels showVisual showForceFields showInteractionForceFields showWireframe')

    root.addObject('DefaultAnimationLoop', )
    root.addObject('CollisionPipeline', depth="6", verbose="0", draw="0")
    root.addObject('BruteForceBroadPhase', )
    root.addObject('BVHNarrowPhase', )
    root.addObject('NewProximityIntersection', name="Proximity", alarmDistance="0.5", contactDistance="0.2")
    root.addObject('CollisionResponse', name="Response", response="PenalityContactForceField")

    # FEM的设置
    # root.addObject('FreeMotionAnimationLoop')
    # root.addObject('GenericConstraintSolver', tolerance=1e-9, maxIterations=200)

    # root.addObject('CollisionPipeline', name='Pipeline', verbose='0')
    # root.addObject('BruteForceBroadPhase', name='BroadPhase')
    # root.addObject('BVHNarrowPhase', name='NarrowPhase')
    # root.addObject('CollisionResponse', name='Response', response='PenalityContactResponse')
    # root.addObject('MinProximityIntersection', name='Proximity', alarmDistance=0.8, contactDistance=0.5)

    obj = root.addChild('object')
    # Rayleigh阻尼影响了软体振动
    obj.addObject('EulerImplicitSolver', name='odesolver', rayleighStiffness='0.1', rayleighMass='0.1')
    obj.addObject('CGLinearSolver', name='linearsolver', iterations='200', tolerance='1.e-9', threshold='1.e-9')

    # obj.addObject('MeshVTKLoader', name='loader', filename='trian.vtk', scale='1', flipNormals='0')
    obj.addObject('MeshGmshLoader', name='loader', filename=f'{file_path}', scale='1', flipNormals='0')
    obj.addObject('MechanicalObject', src='@loader', name='dofs', template='Vec3', translation2=[0., 0., 0.], scale3d=[1.]*3)
    obj.addObject('TriangleSetTopologyContainer', src='@loader', name='container')
    obj.addObject('TriangleSetTopologyModifier', name='modifier')
    obj.addObject('TriangleSetGeometryAlgorithms', name='geomalgo')#, tempate='Vec3')
    obj.addObject('DiagonalMass', name='mass', totalMass='0.1')#, massDensity='0.1')

    X_EPS = 5.e-3
    obj.addObject('BoxROI', name='box', box=f"-0.1 {-X_EPS} -0.1 0.11 {X_EPS} 0.1")
    obj_fixed = obj.addObject('FixedConstraint', name='fixed', indices='@box.indices')

    # 大致相当于E=6.5e3的材料参数
    obj.addObject('MeshSpringForceField', name="springs", trianglesStiffness=90, trianglesDamping=0.3)
    # obj.addObject('TriangularFEMForceField', name='FEM', youngModulus='5.e6', poissonRatio='0.3', method='large')
    obj.addObject('TriangleCollisionModel')
    # obj.addObject('UncoupledConstraintCorrection', defaultCompliance="0.001")

    # Need change the indices to be equal with manipualtion index ######################################################
    # obj.addObject('LinearMovementConstraint', name='cnt1', template="Vec3", indices=[10])
    # obj.addObject('LinearMovementConstraint', name='cnt2', template="Vec3", indices=[11])

    obj_move_list = []
    for q_i in contact:
        obj_move_list.append(obj.addObject('LinearMovementConstraint', name='cnt'+str(q_i), template="Vec3", indices=[q_i]))

    # 输出Sofa设置信息
    # Sofa.msg_info("Scene", f"Contact indices: {obj_linear_move.indices.value}")
    # Sofa.msg_info("User", f"Fixed indices: {obj_fixed.indices.value}")

    return obj, obj_move_list

if __name__ == "__main__":
    nodes, edges, faces = mesh_obj_tri([0.1, 0.1], 0.01)
    write_mshv2_triangular(MESH_DIR / "rectangle.msh", nodes, faces)

    fix = list(range(0, 11))
    contact = [77] # 77, 115, 120

    root = Sofa.Core.Node('root')
    _, move_handle = createScene(root, MESH_DIR / "rectangle.msh", contact)
    Sofa.Simulation.init(root)

    dt = root.dt.value
    obj = root.getChild('object')
    dofs = obj.getObject('dofs')

    sofa_pos_tmp = dofs.findData('position').value
    write_mshv2_triangular(OUTPUT_DIR / f"sofa_contact{contact[0]:d}_step{0:03d}.msh", sofa_pos_tmp, faces)

    for step in range(20):
        add_move(move_handle, dt, np.repeat(np.array([[-1., 0., 0.]]), len(contact), axis=0)*0.001) # 步长需要手动计算
        Sofa.Simulation.animate(root, dt)

        for substep in range(19):
            # sofa_pos_tmp = dofs.findData('position').value
            add_move(move_handle, dt, np.repeat(np.array([[0., 0., 0.]]), len(contact), axis=0))
            # write_vtu(MESH_DIR / "shape.msh", sofa_pos_tmp, OUTPUT_DIR / f"sofa_contact{contact[0]}_step{step:03d}_substep{substep:02d}.vtu")
            Sofa.Simulation.animate(root, dt)

        sofa_pos_tmp = dofs.findData('position').value
        # 正则表达式匹配：regex_pattern = re.compile(r"sofa_contact(\w+)_step(\d{3})\.msh")
        write_mshv2_triangular(OUTPUT_DIR / f"sofa_contact{contact[0]:d}_step{step+1:03d}.msh", sofa_pos_tmp, faces)
        # write_vtu(MESH_DIR / "shape.msh", sofa_pos_tmp, OUTPUT_DIR / f"sofa_contact{contact[0]}_step{step:03d}.vtu")