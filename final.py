import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pickle

np_rand = np.random.rand


def show(data, max_length=10000, stride=1):
    value = data['process']
    title = data['title']
    value = np.array(value)
    x_data = value[:max_length:stride, 0]
    y_data = value[:max_length:stride, 1]
    last_values = 0
    for i in range(len(y_data)):
        y_data[i] = np.log10(y_data[i]) if y_data[i] != 0 else last_values
        last_values = y_data[i]

    plt.plot(x_data, y_data, linewidth=1, linestyle=':')
    plt.title(title)
    plt.ylabel('Benchmark function value(10^x)')
    plt.xlabel('Number of iterations')
    with open('img/%s_%s_%s.png' % (title, stride, max_length), 'wb') as f:
        plt.savefig(f)
    plt.show()


def problem(p):  # 基础算法，用于被社会蜘蛛算法调用求解优解
    x = np.sum((p-56)**2, 1)
    y = np.sum((p-2)**2, 1)
    return x*y


class BenchmarkFunction(object):
    # 基准测试函数
    # 用于测试优化算法在不同问题下的优化能力
    # 以下十个函数是SSA原论文后面列举的基准测试函数中选取的前十个
    # 以下基准函数部分可以收敛大最优解，但是部分无法收敛到最优解
    # 可能是代码编写的基准函数没有写好，也有可能是SSA代码有问题4

    p = 5
    # 表示目标位置。如果优化算法生效，那么将收敛到[5,5,5,5,5....]附近

    @classmethod
    def sphere_function(cls, x):
        o = np.ones((len(x), len(x[0])))*cls.p
        z = x-o
        return np.sum(z**2, 1)

    @classmethod
    def schwefels_problem(cls, x):
        o = np.ones((len(x), len(x[0])))*cls.p
        z = (x - o)*10/100
        mult = 1.
        for i in range(len(z[0])):
            mult = mult*abs(z[:, i])
        return np.sum(np.abs(z), 1) + mult

    @classmethod
    def cigar_function(cls, x):
        o = np.ones((len(x), len(x[0])))*cls.p
        z = x - o
        s = z[:, 0]**2 + (10**6 * np.sum(z[:, 1:]**2, 1))
        return s

    @classmethod
    def discus_function(cls, x):
        o = np.ones((len(x), len(x[0])))*cls.p
        z = x - o
        s = 10**6*z[:, 0]**2 + np.sum(z[:, 1:]**2, 1)
        return s

    @classmethod
    def quadratic_function_noise(cls, x):
        o = np.ones((len(x), len(x[0])))*cls.p
        z = (x-o) * 1.28/100
        for i in range(len(z[0])):
            z[:, i] = (i+1)*z[:, i]**4
        s = np.sum(z, 1) + np_rand(len(x))
        return s

    @classmethod
    def rastrigin_function(cls, x):
        o = np.ones((len(x), len(x[0]))) * cls.p
        z = (x - o) * 5.12 / 100
        s = z ** 2 - 10 * np.cos(2. * np.pi * z) + 10
        return np.sum(s, 1)


    @classmethod
    def ackley_function(cls, x):
        o = np.ones((len(x), len(x[0])))*cls.p
        z = (x - o) * 32 / 100
        s = -20 * np.exp(-0.2*np.sqrt(np.mean(z**2, 1))) - np.exp(np.mean(np.cos(2*np.pi*z), 1)) + 20 + np.e
        return s

    @classmethod
    def griewank_function(cls, x):
        o = np.ones((len(x), len(x[0]))) * cls.p
        z = (x - o) * 600 / 100
        mult = 1.
        zcopy = z.copy()
        for i in range(len(zcopy[0])):
            zcopy[:, i] = zcopy[:, i] / np.sqrt(i + 1)
            mult = mult * np.cos(zcopy[:, i])
        s = 1. / 4000 * np.sum(z ** 2, 1) - mult + 1
        return s





class SSA(object):
    def __init__(self, question, max_bound, min_bound, spider_num=20, dimension=2, max_iteration=10000):
        # question: 优化问题，返回适应度，也就是与最优解的差距，越接近0越好
        # max_bound: 解的最大边界，表示n个维度每个维度的最大值
        # min_bound: 解的最小边界，表示n个维度每个维度的最小值
        # spider_num: 蜘蛛数目，蜘蛛数目越多，算法运行时间越久，最优解的精度也越高
        # dimension: 维度，指的是问题的解的维度，映射到蜘蛛上则是蜘蛛的位置的维度
        # max_iteration: 最大迭代次数，SSA算法只能无限趋近最优解，所以需要添加限定条件来使得算法在某一时刻结束

        self.question = question
        self.dimension = dimension
        self.max_bound = max_bound
        self.min_bound = min_bound
        self.max_iteration = max_iteration
        self.spider_num = spider_num
       
        self.best_score = np.Inf  # 当前最优解的得分（也叫适应度）
        self.bert_score_history = []  # 历史最优解得分
        self.best_position = np.zeros(self.dimension)  # 历史最优解数组，每次得到一个更优的解，那么记录下来
        self.sp_position = None  # 蜘蛛当前位置，n个蜘蛛每个蜘蛛m个维度
        self.vibrations = None  # 蜘蛛当前振动，也就是通过在当前位置求解的适应度生成的振动，适应度越小，振动越大
        self.cs = None  # 蜘蛛的惰性数组，表示蜘蛛多久没有改变过目标位置了

    def run(self):
        rs = {}
        rs['title'] = self.question.__name__
        rs['process'] = []
        rs['positions'] = []
        print("求解问题：%s" % self.question.__name__)
        self.sp_position = np_rand(self.spider_num, self.dimension) * (self.max_bound-self.min_bound) + self.min_bound
        # 由解的边界约束确定初始随机位置（也就是初始解)返回的是一个多维数组,(spider_num,self.dimension)(25,30)也就是25行30列的数据，表示25个蜘蛛，每个蜘蛛的解有30个变量
        target_position = self.sp_position.copy()  # 蜘蛛的目标位置，蜘蛛会向目标位置随机走，初始时复制当前的位置，
        target_vibrations = np.zeros(self.spider_num)  # 蜘蛛的目标位置的振动，用于比较接收到的最大振动和目标振动来选择是否将目标位置改成接收最大振动的位置
        mask = np.zeros((self.spider_num, self.dimension))  # 掩码，表示蜘蛛目标位置的组成boolean数组，这个数组将当前目标数组和随机目标数组组合成最终的目标数组
        move = np.zeros((self.spider_num, self.dimension))  # 上一次蜘蛛的移动位移，蜘蛛下一次位移由上一次位移的随机一部分和距离目标位置的距离的随机一部分组成
        self.cs = np.zeros(self.spider_num)  # 蜘蛛的惰性数组，数值越大表示蜘蛛越久没有改变过目标位置，所以需要在mask中随机位置的分量
        r_a = 1.  # 振动衰减系数 这个数越大表示衰减越小，也就是距离对于振动衰减的影响更小
        p_c = 0.7  # 表示蜘蛛的对应掩码是否需要改变的概率相关系数， 改变概率为 1-pc^cs 也就是cs越大改变概率越大。pc决定cs的影响力，pc越小时1-pc^cs能更快的趋近0
        p_m = 0.3  # 表示掩码每个值改变的概率，有p_c的概率变成1，1-p_c的概率变成0。代码中用小于号表示 随机数 0.5<0.7 true, 随机数0.8>0.7,也就是在[0, 0.7]的随机数为真

        iteration = 0  # 初始化迭代系数
        while iteration < self.max_iteration:  # 开始迭代

            spider_fitness = self.question(self.sp_position)  # 蜘蛛的在当前位置求解的适应度，适应度越低表示越好
            rs['process'].append([iteration, np.min(spider_fitness)])
            rs['positions'].append([iteration, self.sp_position])
            if iteration % 1000 == 0:
                print(iteration)
            iteration += 1  # 迭代系数加一
            # 每个蜘蛛的维度的标准差，这个标准差用于衰减系数的计算
            std_mean = 0.
            # 对于每个维度
            for i in range(self.dimension):
                dim = 0.
                # 求第i个维度在n个蜘蛛中的平均值
                for p in self.sp_position:
                    # 计算出这个维度在所有蜘蛛的总和
                    dim = dim + p[i]
                # 求出这个维度在n个蜘蛛的平均值
                dim_mean = dim / self.spider_num
                std = 0.
                # 根据已知的第i个维度的平均值求这个维度的标准差
                for p in self.sp_position:
                    std = std + (p[i] - dim_mean) ** 2
                # 求m个维度的标准差总和
                std_mean = std_mean + np.sqrt(std / self.spider_num)
            # 对总和标准差进行平均
            std_mean = (std_mean / self.dimension) if (std_mean / self.dimension) != 0 else 1
            distance = []
            # 计算n个点之间互相的曼哈顿距离，由一个一维数组生成二维数组，行列表示两个位置，值表示距离
            # 曼哈顿距离：（X1, Y1） （X2, Y2） 曼哈顿距离 = |X1-X2|+|Y1-Y2|
            for p in self.sp_position:
                d = []
                for other_p in self.sp_position:
                    dist = 0
                    # 累加m个维度上的差值绝对值总和
                    for i in range(len(p)):
                        dist = dist + abs(p[i] - other_p[i])
                    d.append(dist)
                distance.append(d)
            # 将distance变成np的array，以便进行数学运算
            distance = np.array(distance)
            if np.min(spider_fitness) < self.best_score:  # 如果当前所有蜘蛛的求解的适应度小于当前的最优适应度，说明找到一个更好的解
                self.best_score = np.min(spider_fitness)  # 那么记录下这个最小的适应度
                self.best_position = self.sp_position[np.argmin(spider_fitness)].copy()  # 复制这个当前最小适应度的解
            self.bert_score_history.append(self.best_score)  # 将当前最小的适应度记录到历史数组中

            vibrations_source = np.log(1. / (spider_fitness - (-1E-100)) + 1)  # 计算振动源大小，-1E-100表示一个极小的值，一个比最小解要小的值
            x = -distance / (std_mean * r_a)
            attenuation = np.exp(x)  # 计算出距离衰减系数
            # 根据振动源数组和振动衰减二维矩阵生成接收振动矩阵,
            # 接收矩阵中，行表示接收振动的蜘蛛，列表示振动源的蜘蛛
            # [1,2,3] * [[0.4, 0.5, 0.1], = [[1,2,3], * [[0.4, 0.5, 0.1] = 对位元素相乘 = [[0.4,1.,0.3],
            #            [0.2, 0.4, 0.3],    [1,2,3]     [0.2, 0.4, 0.3],                 [0.2,0.8,0.9],
            #            [0.1, 0.2, 0.4]]    [1,2,3]]    [0.1, 0.2, 0.4]]                 0.1,0.4,1.2]]
            td_vibrations = np.zeros([self.spider_num, self.spider_num])  # 构建一个二维的振动数组，用于计算接收振动
            for i in range(self.spider_num):
                td_vibrations[i] = vibrations_source  # 每一行都是振动源，使得维度相同，便于和衰减矩阵进行运算
            # 根据构建的二维振动矩阵和衰减矩阵计算接收振动矩阵
            receiver_vibrations = td_vibrations * attenuation
            # 计算每个蜘蛛接收振动中最大振动的索引
            max_receiver_index = []
            # 遍历n个蜘蛛接收到的n个振动
            for v in receiver_vibrations:
                max_index = v.argmax()  # 找出最大振动的索引
                max_receiver_index.append(max_index)  # 记录下这个索引
            max_index = np.array(max_receiver_index)  # 将python的数组变成np的数组，便于计算
            keep_target = np.zeros(self.spider_num)  # 这个数组用于表示每个蜘蛛是否需要保持目标位置目标振动不变
            for i in range(self.spider_num):
                # 如果这蜘蛛接收到的振动中的最大振动小于等于上一次的目标振动，那么对应的数值设为1，表示不需要改变
                if receiver_vibrations[i, max_index[i]] <= target_vibrations[i]:
                    keep_target[i] = 1
            # 目标选择矩阵，表示新的目标位置数组怎么由上一次的目标位置数组和接收振动数组组成
            target_choose_matrix = np.zeros(self.spider_num * self.dimension)
            # 对keep_target中spider_num个元素每个都重复dimension次,使得原来[1,2,3]变成 [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3]
            for i in range(self.spider_num):
                # 对每个维度进行遍历
                for j in range(self.dimension):
                    target_choose_matrix[i*self.dimension+j] = keep_target[i]
            # 再对数组进行结构调整，使得每一行都是一样的数，这个矩阵中每一行都表示一个蜘蛛m个维度是否保持的信息，一个蜘蛛如果要保持，那么所有的维度都要保持，所以要重复一行
            target_choose_matrix = target_choose_matrix.reshape(self.spider_num, self.dimension)
            # 根据是否保持数组来更新cs，如果是否保持数组中的值为1，那么cs加一，否则清0
            self.cs = self.cs * keep_target + keep_target
            for i in range(self.spider_num):
                # 如果keep_target[i] >= 1那么cs加一，否则清0
                self.cs[i] = self.cs[i]+1 if keep_target[i] >= 1 else 0
            # 最大接收振动矩阵
            max_receiver_vibrations = np.zeros(self.spider_num)
            for i in range(self.spider_num):
                # 根据之前计算的最大振动索引获取一个蜘蛛接收到n个振动中的最大振动
                max_receiver_vibrations[i] = receiver_vibrations[i, max_index[i]]
            # 根据上一次最大振动，保持数组和接收的最大振动数组来构成新的目标振动
            target_vibrations = target_vibrations * keep_target + max_receiver_vibrations * (1 - keep_target)
            # 最大接收振动对应的位置
            max_receiver_position = np.zeros((self.spider_num, self.dimension))
            for i in range(self.spider_num):
                # 从所有蜘蛛位置中找到最大接收振动的位置
                max_receiver_position[i] = self.sp_position[max_index[i]]
            # 根据上一次目标位置、保持矩阵和最大接收振动对应的位置生成新的目标位置
            target_position = target_position * target_choose_matrix + max_receiver_position * (1 - target_choose_matrix)
            # 复制当前所有蜘蛛位置，用于随机变换，也就是随机两两蜘蛛交换位置
            rand_position = self.sp_position.copy()
            # 对蜘蛛进行随机位置交换
            for i in range(self.spider_num):
                j = np.random.randint(low=0, high=self.spider_num)
                # 随机两两蜘蛛交换位置
                rand_position[i], rand_position[j] = rand_position[j], rand_position[i]
            # 生成一个随机概率的掩码数组，这个数组将和原来的掩码数组根据随机概率进行组合
            new_mask = np.ceil(np_rand(self.spider_num, self.dimension) + np_rand() * p_m - 1)
            # 保持掩码，表示原掩码中哪些行需要保持不变，哪些行需要变成随机掩码数组的行
            keep_mask = np_rand(self.spider_num) < p_c ** self.cs
            # 将保持掩码进行重复扩展为保持掩码矩阵，使得原来[1,0,1]变成 [[1,1,1],[0,0,0],[1,1,1]]
            keep_mask_matrix = np.repeat(keep_mask, self.dimension).reshape(self.spider_num, self.dimension)
            # 根据原掩码，保持掩码矩阵和随机掩码矩阵生成新的掩码
            mask = keep_mask_matrix * mask + (1 - keep_mask_matrix) * new_mask
            # 根据原位置数组、新掩码、目标位置数组生成新的位置数组
            new_position = rand_position * mask + target_position * (1 - mask)
            # 根据新的位置数组和当前位置数组计算出下一步位移
            next_move = new_position - self.sp_position
            # 生成下一步位移的随机系数
            rand_r = np_rand(self.spider_num, self.dimension)
            # 根据上一次位移和它的随机系数+下一次位移和下一步位移的随机系数组成最终位移
            next_move = move * np.repeat(np_rand(self.spider_num), self.dimension).reshape(self.spider_num, self.dimension) + next_move * rand_r
            # 根据最终位移进行位置更新
            self.sp_position = self.sp_position + next_move
            # 记录下这一次的位移
            move = next_move
            # 违反约束处理。。。未完成
        rs.update({'global_best_fitness': self.best_score,
                'global_best_solution': self.best_position,
                'sp_position': self.sp_position,
                'iterations': iteration + 1})
        with open('data_position/'+self.question.__name__+'.txt', 'wb') as f:
            pickle.dump(rs, f, True)
        return rs


if __name__ == '__main__':
    class Logger(object):
        def __init__(self, filename="log.txt"):
            self.terminal = sys.stdout
            self.log = open(filename, "a")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass



    max_bound = np.array([5.5]).repeat(2)
    min_bound = np.array([4.5]).repeat(2)
    show((SSA(BenchmarkFunction.sphere_function, max_bound=max_bound, min_bound=min_bound, max_iteration=1000).run()))

    # show(SSA(BenchmarkFunction.schwefels_problem, max_bound=max_bound, min_bound=min_bound, max_iteration=1000).run())
    # show(SSA(BenchmarkFunction.cigar_function, max_bound=max_bound, min_bound=min_bound, max_iteration=1000).run())
    # show(SSA(BenchmarkFunction.discus_function, max_bound=max_bound, min_bound=min_bound, max_iteration=1000).run())
    # show(SSA(BenchmarkFunction.quadratic_function_noise, max_bound=max_bound, min_bound=min_bound, max_iteration=1000).run())
    # show(SSA(BenchmarkFunction.rastrigin_function, max_bound=max_bound, min_bound=min_bound, max_iteration=1000).run())
    # show(SSA(BenchmarkFunction.ackley_function, max_bound=max_bound, min_bound=min_bound, max_iteration=1000).run())
    # show(SSA(BenchmarkFunction.griewank_function, max_bound=max_bound, min_bound=min_bound, max_iteration=1000).run())


