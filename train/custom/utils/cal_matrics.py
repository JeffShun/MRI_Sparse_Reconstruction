import os
import re
import SimpleITK as sitk
import numpy as np
import argparse
import pandas as pd
from multiprocessing import Pool
import sys

def parse_args():
    parser = argparse.ArgumentParser('cal precision')
    parser.add_argument('--pred_path',
                        default='../../../example/data/output_test',
                        type=str)
    parser.add_argument('--label_path',
                        default='../../../example/data/input_test/label',
                        type=str)
    parser.add_argument('--output_path',
                        default='../../../example/data/output_test.csv',
                        type=str)
    parser.add_argument('--dist_threshold',
                        default=10.0,
                        type=float)
    parser.add_argument('--angle_error_threshold',
                        default=10.0,
                        type=float)
    parser.add_argument('--print_path',
                        default='',
                        type=str)
    args = parser.parse_args()
    return args


def cal_shift_error(pred_img, label_img):
    spacing = pred_img.GetSpacing()
    rotate_axis = [2,1,0] #rotate the spacing from x,y,z to z,y,x
    spacing = np.array(spacing)[rotate_axis]
    pred_arr = (sitk.GetArrayFromImage(pred_img)).astype("uint8")
    label_arr = (sitk.GetArrayFromImage(label_img)).astype("uint8")
    shift_errors_abs = []
    shift_errors_ret = []
    for i in range(1, 6):
        pred = np.array(list(zip(*np.where(pred_arr==i))))
        pred_center = np.mean(pred, 0)
        pred_center = tuple((pred_center + 0.5).astype(np.int64))
        pred_center_phy = np.array(pred_center) * spacing 

        label = np.array(list(zip(*np.where(label_arr==i))))
        label_center = np.mean(label, 0)
        label_center = tuple((label_center + 0.5).astype(np.int64))
        label_center_phy = np.array(label_center) * spacing         
        shift = np.sqrt(np.sum((pred_center_phy - label_center_phy)**2))
        shift_errors_abs.append(shift)
        denominator = np.sqrt(np.sum((np.array(pred_arr.shape)*spacing)**2))
        shift_errors_ret.append(shift/denominator)
    return shift_errors_abs, shift_errors_ret


def cal_normal_vector(image):
    point_img = sitk.GetArrayFromImage(image)
    points = []
    for i in range(1,6): 
        loc = np.array(list(zip(*np.where(point_img==i))))
        loc_center = np.mean(loc, 0)
        loc_center = tuple((loc_center + 0.5).astype(np.int64))
        points.append(loc_center)

    # 获取图像信息
    size = image.GetSize()  # 图像尺寸
    spacing = image.GetSpacing()  # 间距
    origin = image.GetOrigin()  # 原点坐标

    vtkpoints = []
    for point in points:
        pixel_z,pixel_y,pixel_x = point
        # 将像素坐标转换为 VTK 坐标
        vtk_x = origin[0] + pixel_x * spacing[0]
        vtk_y = origin[1] + pixel_y * spacing[1]
        vtk_z = origin[2] + pixel_z * spacing[2]
        vtk_point = (vtk_x,vtk_y,vtk_z)
        vtkpoints.append(vtk_point)

    point1 = vtkpoints[0]  
    point2 = vtkpoints[1]  
    point3 = vtkpoints[2] 
    point4 = vtkpoints[3]  
    point5 = vtkpoints[4]   

    # 计算矢状面的法线向量
    v1 = np.array(point5) - np.array(point1)
    v1 /= np.linalg.norm(v1)
    v2 = np.array(point3) - np.array(point1)
    v2 /= np.linalg.norm(v2)
    normal_sagittal = tuple(np.cross(v1, v2))
    normal_sagittal /= np.linalg.norm(normal_sagittal)

    # 计算横断面的法线向量
    v1 = np.array(point5) - np.array(point4)
    v1 /= np.linalg.norm(v1)
    axis1_axial = v1
    axis2_axial = normal_sagittal
    normal_axial = tuple(np.cross(axis1_axial, axis2_axial))
    normal_axial /= np.linalg.norm(normal_axial)

    # 计算冠状面的法线向量
    v1 = np.array(point3) - np.array(point2)
    v1 /= np.linalg.norm(v1)
    axis1_coronal = v1
    axis2_coronal = normal_sagittal
    normal_coronal = tuple(np.cross(axis1_coronal, axis2_coronal))
    normal_coronal /= np.linalg.norm(normal_coronal)   

    return normal_sagittal, normal_axial, normal_coronal

def cal_angle_error(pred_img, label_img):
    
    normal_sagittal_pred, normal_axial_pred, normal_coronal_pred = cal_normal_vector(pred_img)
    normal_sagittal_label, normal_axial_label, normal_coronal_label = cal_normal_vector(label_img)

    iproduct_sagittal = np.clip(np.dot(np.array(normal_sagittal_pred), np.array(normal_sagittal_label)),0,1)
    sagittal_angle_error = 180*np.arccos(iproduct_sagittal)/np.pi
    iproduct_axial = np.clip(np.dot(np.array(normal_axial_pred), np.array(normal_axial_label)),0,1)
    axial_angle_error = 180*np.arccos(iproduct_axial)/np.pi
    iproduct_coronal = np.clip(np.dot(np.array(normal_coronal_pred), np.array(normal_coronal_label)),0,1)
    coronal_angle_error = 180*np.arccos(iproduct_coronal)/np.pi
    return [sagittal_angle_error, axial_angle_error, coronal_angle_error]


def multiprocess_pipe(input):
    p_f, l_f = input
    pred_img = sitk.ReadImage(p_f)
    label_img = sitk.ReadImage(l_f)
    shift_error_abs, shift_error_ret = cal_shift_error(pred_img, label_img)
    angle_error = cal_angle_error(pred_img, label_img)
    return shift_error_abs, shift_error_ret, angle_error


if __name__ == "__main__":
    args = parse_args()
    dist_threshold = args.dist_threshold
    angle_error_threshold = args.angle_error_threshold
    label_path = args.label_path
    pred_path = args.pred_path
    print_path = args.print_path
    output_path = args.output_path
    output_dir = "/".join(output_path.split("/")[:-1])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pids = os.listdir(pred_path)
    pool = Pool(8)
    inputs = []   
    for pid in pids:
        p_f = os.path.join(pred_path, pid, pid+".seg.nii.gz")
        l_f = os.path.join(label_path, pid+".seg.nii.gz")
        inputs.append((p_f, l_f))
    result = pool.map(multiprocess_pipe, inputs)
    pool.close()
    pool.join()

    right_count = 0
    for shift_error_abs, shift_error_ret, angle_error in result:
        if (np.array(shift_error_abs) < dist_threshold).all() and (np.array(angle_error) < angle_error_threshold).all():
            right_count+=1
    if print_path != "":
        f = open(print_path, 'a+')  
        print("Total: %d Fail: %d 合格率: %.3f"%(len(result), len(result)-right_count, right_count/len(result)), file=f)
        f.close()
    print("Total: %d Fail: %d 合格率: %.3f"%(len(result), len(result)-right_count, right_count/len(result)))
    p1_error_abs = [sample[0][0] for sample in result]
    p2_error_abs = [sample[0][1] for sample in result]
    p3_error_abs = [sample[0][2] for sample in result]
    p4_error_abs = [sample[0][3] for sample in result]
    p5_error_abs = [sample[0][4] for sample in result]

    p1_error_ret = [sample[1][0] for sample in result]
    p2_error_ret = [sample[1][1] for sample in result]
    p3_error_ret = [sample[1][2] for sample in result]
    p4_error_ret = [sample[1][3] for sample in result]
    p5_error_ret = [sample[1][4] for sample in result]

    angle_error_1 = [sample[2][0] for sample in result]
    angle_error_2 = [sample[2][1] for sample in result]
    angle_error_3 = [sample[2][2] for sample in result]

    res = pd.DataFrame(np.array([pids,p1_error_abs,p2_error_abs,p3_error_abs,p4_error_abs,p5_error_abs,p1_error_ret,p2_error_ret,p3_error_ret,p4_error_ret,p5_error_ret,angle_error_1,angle_error_2,angle_error_3]).T)
    res.to_csv(output_path,index=False,header=["pid","p1_error_abs","p2_error_abs","p3_error_abs","p4_error_abs","p5_error_abs",
                                            "p1_error_ret","p2_error_ret","p3_error_ret","p4_error_ret","p5_error_ret",
                                            "sagittal_angle_error","axial_angle_error","coronal_angle_error"])
