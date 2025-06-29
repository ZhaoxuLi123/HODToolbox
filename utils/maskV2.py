import numpy as np
import random


def point_on_square_edges(length):
    semi = int(length//2)
    point_list = [(semi,i) for i in range(-semi,semi+1)] + [(-semi,i) for i in range(-semi,semi+1)]+\
                 [(i, semi) for i in range(-semi+1, semi)] + [(i,-semi) for i in range(-semi+1, semi)]
    return point_list

def split_integer(m, n):
    assert n > 0
    quotient = int(m / n)
    remainder = m % n
    if remainder > 0:
        return [quotient] * (n - remainder) + [quotient + 1] * remainder
    if remainder < 0:
        return [quotient - 1] * -remainder + [quotient] * (n + remainder)
    return [quotient] * n

def find_nearest_point(sub_list,cal_list):
    ratio = np.zeros(len(cal_list))
    for id in range(len(cal_list)):
        point =cal_list[id]
        connect_8point = [(point[0] - 1, point[1]), (point[0] + 1, point[1]), (point[0], point[1] - 1),
                          (point[0], point[1] + 1), (point[0]-1, point[1]-1), (point[0]+1, point[1]-1),
                          (point[0]-1, point[1]+1), (point[0]+1, point[1]+1)]

        n1=len(set(connect_8point) & set(sub_list))
        n2 =len(set(connect_8point) & set(cal_list))
        if (n1 + n2)>0:
            ratio[id] = n1/(n1+n2)
        else:
            ratio[id] = 0
    nearest_ids = list(np.where(ratio == np.max(ratio))[0])
    nearest_points =[ cal_list[id] for id in nearest_ids]
    return nearest_points


def judge_adjacent( img):
    adj_point_list =[]
    first_list = []
    move_list = [(-1,0),(1,0) , (0, -1), (0, 1)]
    m, n = img.shape
    for i in range(m):
        for j in range(n):
            if img[i,j]>0:
                for move in move_list:
                    if 0 <= i+move[0] <= m-1 and 0 <= j + move[1] <= n-1:
                        if img[i+move[0],j + move[1]] ==0:
                            if (i+move[0],j + move[1]) not in adj_point_list:
                                adj_point_list = adj_point_list + [(i+move[0],j + move[1])]
    for point in adj_point_list:
        adj_num = 0
        for move in move_list:
            if img[point[0] + move[0], point[1]  + move[1]] >0:
                adj_num += 1
        if adj_num>2:
            first_list.append(point)
            adj_point_list.remove(point)
    return first_list, adj_point_list


def generate_single_random_shape(area):
    img =np.zeros([19,19])
    point_num =1
    img[9,9]=1
    point_list = [(0,0)]
    while point_num<area:
        first_list, adj_point_list = judge_adjacent(img)
        if len(first_list)>0:
            for point in first_list:
                if  point_num< area:
                    img[point] = 1
                    point_list = point_list + [(point[0]-9,point[1]-9)]
                    point_num += 1
        if len(adj_point_list) > 0:
            for  point in adj_point_list:
                if random.random() < 0.5 and point_num<area:
                    img[point] = 1
                    point_list = point_list + [(point[0]-9,point[1]-9)]
                    point_num += 1
    return point_list



class MaskGenerator(object):
    def __init__(self, w, h, target_num_max, target_list,
                 size_dict={0:(1, 1), 1:(1, 2), 2:(3,5), 3:(6,10), 4:(11,16)},
                 edge_abu_interval=(0.01, 1),
                 center_edge_abu_interval=(0.2, 1),
                 single_pixel_abu_interval=(0.05, 0.2),
                 object_cat='random'
                 ):
        assert object_cat in ['random', 'in_turn'], 'object_cat should be random or in turn'
        self.w = w
        self.h = h
        self.target_num_max = target_num_max
        self.target_list = target_list
        self.size_dict = size_dict
        self.edge_abu_interval = edge_abu_interval
        self.center_edge_abu_interval = center_edge_abu_interval
        self.single_pixel_abu_interval = single_pixel_abu_interval
        self.object_cat = object_cat
        self.first_layer = [(0, 0)]
        self.second_layer = point_on_square_edges(3)
        self.third_layer = point_on_square_edges(5)

    def single_random_shape(self, area):
        # if area>9:
        #     choice_list = [self.third_layer[p]  for p in random.sample(list(range(len(self.third_layer))), area-9)]
        #     all_list = self.first_layer+ self.second_layer+choice_list
        # elif area>1:
        #     choice_list = [self.second_layer[p]  for p in random.sample(list(range(len(self.second_layer))), area-1)]
        #     all_list = self.first_layer+choice_list
        # else:
        #     all_list = self.first_layer
        all_list = generate_single_random_shape(area)
        point_list = []
        edge_point_list = []
        for point in all_list:
            connect_point = [(point[0] - 1, point[1]), (point[0] + 1, point[1]), (point[0], point[1] - 1),
                          (point[0], point[1] + 1)]
            if set(connect_point).issubset(set(all_list)):
                point_list = point_list + [point]
            else:
                edge_point_list = edge_point_list + [point]
        # elif area>1:
        #     choice_list = [self.second_layer[p]  for p in random.sample(list(range(len(self.second_layer))), area-1)]
        #     point_list = self.first_layer
        #     edge_point_list =choice_list
        # else:
        #     point_list =self.first_layer
        #     edge_point_list = []
        return all_list, point_list, edge_point_list

    def sort_edge_point(self,all_point_list, edge_point_list):
        all_point = np.array(all_point_list)
        center_point = np.mean(all_point,axis=0)
        edge_point_dis_list = [(point_i[0]-center_point[0]) ** 2 + (point_i[1]-center_point[0]) ** 2 for point_i in edge_point_list]
        edge_point_dis_sort = np.argsort(np.array(edge_point_dis_list))
        edge_point_dis_sort_id = np.zeros(len(edge_point_list))
        for ep_i, dis_sort_id in enumerate(edge_point_dis_sort):
            edge_point_dis_sort_id[dis_sort_id] = ep_i
        return edge_point_dis_sort_id


    def inimage(self,center_point, point):
        if ((center_point[0]+point[0]) < self.h) and ((center_point[1]+point[1]) < self.w):
            if ((center_point[0] + point[0]) >=0) and ((center_point[1] + point[1]) >= 0):
                return True
        else:
            return False

    def remote_out_range(self,center_point, point_list):
        new_list = []
        for point in point_list:
            if self.inimage(center_point, point):
                new_list.append(point)
        return new_list


    def __call__(self, ):
        target_mask = np.zeros([self.w, self.h])  # 目标实例标注
        cat_mask = np.zeros([ self.w,  self.h])  # 逐光谱的目标类别标注
        abundance_masks = np.zeros([self.w, self.h, 4])  # 丰度系数，第一维目标丰度，第二到四维为目标全像元曲线各组成光谱的百分比
        bbox_list = []
        segment_list = []
        area_list = []
        cat_list = []
        target_num = random.randint(1, self.target_num_max) # 添加目标总数
        center_pos = []   # 中心点位置
        for i in range(target_num):
            while 1:
                y = random.randint(3, self.h - 4)
                x = random.randint(3, self.w - 4)
                flag = 0
                for j in range(len(center_pos)):
                    if ((center_pos[j][0]-y)**2+(center_pos[j][1]-x)**2)<64:  # 与所有中心点的l1距离都大于阈值
                        flag = 1
                if flag == 0:
                    break
            center_pos = center_pos+[(y,x)]
            if self.object_cat == 'random':
                t_cat = random.randint(1, len(self.target_list))
            else:
                t_cat = i % len(self.target_list) + 1
            t_info = self.target_list[t_cat-1]
            size = t_info['size']
            cat_list = cat_list + [t_info['cat_id']]
            assert size in self.size_dict.keys(), 'size is not defined'
            area = random.randint(self.size_dict[size][0], self.size_dict[size][1])
            all_point_list, cener_point_list, edge_point_list = self.single_random_shape(area)
            # all_point_list 目标所有的像素点 point_list 全像元像素 edge_point_list 边缘混合像素
            all_point_list = self.remote_out_range((y, x), all_point_list)
            cener_point_list = self.remote_out_range((y, x), cener_point_list)
            edge_point_list = self.remote_out_range((y, x), edge_point_list)  # 判断是否超过图像尺寸
            abund = self.edge_abu_interval[0] + \
                    (self.edge_abu_interval[1]-self.edge_abu_interval[0])*np.random.rand(len(edge_point_list))  # 边缘像元丰度在0.01-1之间
            abund = list(abund)
            abund.sort(reverse=True) # 丰度从大到小
            if len(cener_point_list) == 0:
                abund[0] = self.center_edge_abu_interval[0] + \
                    (self.center_edge_abu_interval[1]-self.center_edge_abu_interval[0])*np.random.rand() #中心像元为边缘像元，确保中心像元丰度大于0.2
            abund.sort(reverse=True)
            if size == 0:   # size 为0均为单像素亚像元目标
                abund = [self.single_pixel_abu_interval[0] + (self.single_pixel_abu_interval[1] - self.single_pixel_abu_interval[0])*np.random.rand()]
            edge_point_dis_sort_id = self.sort_edge_point(all_point_list, edge_point_list)  #按与质心的距离从小到大排列
            if len(cener_point_list) > 0:
                for point in cener_point_list:
                    cat_mask[point[0]+y,point[1]+x] = t_cat
                    abundance_masks[point[0]+y,point[1]+x,0] = 1
                    target_mask[point[0]+y,point[1]+x] = i+1
            for p_id in range(len(edge_point_list)):
                cat_mask[edge_point_list[p_id][0] + y, edge_point_list[p_id][1] + x] = t_cat
                abundance_masks[edge_point_list[p_id][0] + y, edge_point_list[p_id][1] + x, 0] = abund[int(edge_point_dis_sort_id[p_id])]
                target_mask[edge_point_list[p_id][0] + y, edge_point_list[p_id][1] + x] = i+1
            all_points = np.array(all_point_list)
            y1 = float(np.min(all_points[:, 0])+ y)
            x1 = float(np.min(all_points[:, 1])+ x)
            y2 = float(min(np.max(all_points[:, 0])+ y+1,self.h))
            x2 = float(min(np.max(all_points[:, 1])+ x+1,self.w))
            height = float(y2-y1)
            width = float(x2-x1)
            bbox_list = bbox_list + [[x1, y1, width, height]]
            segment_list = segment_list + [[x1, y1, x1, y2, x2, y2, x2, y1]]
            area_list = area_list + [float(height*width)]
            if t_info['mix'] == 'random':
                for p_id in range(len(all_point_list)):
                    sub_abuns = [random.random() for _ in range(t_info['sp_num'])]
                    total_abun = sum(sub_abuns)
                    normalized_sub_abun = [sub_abun / total_abun for sub_abun in sub_abuns]
                    for spec_i in range(t_info['sp_num']):
                        # subcat_mask[all_point_list[p_id][0] + y, all_point_list[p_id][1] + x] = subcat[p_id]
                        abundance_masks[all_point_list[p_id][0] + y, all_point_list[p_id][1] + x, int(spec_i+1)] \
                            = normalized_sub_abun[spec_i]
                    abundance_masks[all_point_list[p_id][0] + y, all_point_list[p_id][1] + x, t_info['sp_num']] = 1- sum(normalized_sub_abun[:spec_i])
            elif t_info['mix'] == 'cascade':
                subcat_nums = split_integer(len(all_point_list),t_info['sp_num'])
                sub1_n = subcat_nums[0]
                sub1_list = [edge_point_list[random.randint(1, len(edge_point_list))-1]]
                all_point_list.remove(sub1_list[0])
                while len(sub1_list)<sub1_n:
                     nearest_points = find_nearest_point(sub1_list,all_point_list)
                     if len(nearest_points) <= (sub1_n -len(sub1_list)):
                         add_points = nearest_points
                     else:
                         sampled_ids = random.sample(list(range(len(nearest_points))), sub1_n - len(sub1_list))
                         add_points = [nearest_points[s_id] for s_id in sampled_ids]
                     sub1_list = sub1_list + add_points
                     for point_i in add_points:
                         all_point_list.remove(point_i)
                for point_i in sub1_list:
                    abundance_masks[point_i[0] + y, point_i[1] + x, 1] = 1
                for point_i in all_point_list:
                    abundance_masks[point_i[0] + y, point_i[1] + x, 2] = 1
                if t_info['sp_num'] == 3:
                    nearest_points = find_nearest_point(sub1_list, all_point_list)
                    sub2_n = subcat_nums[1]
                    sub2_list = [nearest_points[0]]
                    all_point_list.remove(sub2_list[0])
                    while len(sub2_list) < sub2_n:
                        nearest_points = find_nearest_point(sub2_list, all_point_list)
                        if len(nearest_points) <= (sub2_n - len(sub2_list)):
                            add_points = nearest_points
                        else:
                            sampled_ids = random.sample(list(range(len(nearest_points))), sub2_n - len(sub2_list))
                            add_points = [nearest_points[s_id] for s_id in sampled_ids]
                        sub2_list = sub2_list + add_points
                        for point_i in add_points:
                            all_point_list.remove(point_i)
                    for point_i in all_point_list:
                        abundance_masks[point_i[0] + y, point_i[1] + x, 2] = 0
                        abundance_masks[point_i[0] + y, point_i[1] + x, 3] = 1
            else:
                for point_i in all_point_list:
                    abundance_masks[point_i[0] + y, point_i[1] + x, 1] = 1
        return cat_mask, abundance_masks, target_mask, bbox_list, segment_list, area_list, cat_list

