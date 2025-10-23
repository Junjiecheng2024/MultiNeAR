import concurrent.futures  # 标准库：并发执行工具（线程/进程池、Future 等）
import multiprocessing     # 标准库：多进程相关（CPU 核心数、进程等）
import time                # 标准库：时间工具（计时、睡眠）
import warnings            # 标准库：发出运行时警告
from tqdm import tqdm      # 第三方：进度条库 tqdm

AVAILABLE_CORES = multiprocessing.cpu_count()  # 读取当前机器可用的 CPU 核心数

print('Cores available:', AVAILABLE_CORES)  # 打印可用核心数，便于观察环境

class TaskRunner:  # 定义一个“任务运行器”类，用多进程并发执行一组独立任务
    def __init__(self, task, arg_list, max_workers=AVAILABLE_CORES // 2, use_tqdm=True):
        self.max_workers = max_workers  # 并发进程数上限（默认为可用核心数的一半）
        self.task = task                # 需要并发执行的函数/可调用对象，签名形如 task(arg)
        self.arg_list = arg_list        # 任务参数列表/可迭代对象，每个元素传给 task
        self.use_tqdm = use_tqdm        # 是否使用 tqdm 显示进度条

    def run(self):  # 启动并行执行
        self.now = time.time()  # 记录开始时间，用于统计耗时
        # 使用进程池执行任务；with 语句保证池在使用完成后正确关闭/回收
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务到进程池，返回 Future 集合；每个 Future 对应一个任务
            futures = {executor.submit(self._task, arg)
                       for arg in self.arg_list}
            # 返回一个迭代器，按任务完成的先后顺序产生 Future
            completed = concurrent.futures.as_completed(futures)
            if self.use_tqdm:  # 如果启用进度条
                completed = tqdm(completed, total=len(self.arg_list))  # 用 tqdm 包装 completed 迭代器
            # 收集结果：依次取每个已完成 Future 的返回值，保存到 self.results_ 列表
            self.results_ = [future.result() for future in completed]
        print("Finished all @%ss" % (time.time() - self.now))  # 打印总耗时

    def _task(self, arg):  # 进程池中真正执行的包装函数（在子进程里运行）
        try:
            ret = self.task(arg)  # 调用用户传入的任务函数，并获得返回值
        except Exception as e:  # 捕获任务执行中的异常，避免进程崩溃
            warnings.warn("TASK ERROR:=====" + str(e) + "=====" + str(arg))  # 发出警告，包含异常与参数
            return "error", arg, None  # 统一返回错误标记，便于后续统计
        if not self.use_tqdm:  # 未启用 tqdm 时，手动打印每个任务完成的耗时
            print("Finished %s @%ss" % (arg, time.time() - self.now))
        return "success", arg, ret  # 正常返回：状态、对应参数、任务结果

    @property
    def errors_(self):  # 属性：提取所有失败任务对应的参数列表
        if not hasattr(self, "results_"):  # 若还未运行 run()，则没有 results_
            raise AttributeError  # 抛出属性错误提示调用顺序不当
        return [r[1] for r in self.results_ if r[0] == "error"]  # 过滤状态为 error 的项，取其参数 r[1]

# 模块作为脚本直接运行时的示例用法（不会在 import 时执行）
if __name__ == '__main__':
    print("The following provides the usage code of the multi-core `TaskRunner`.")  # 说明文字

    def task(x):          # 定义一个示例任务：打印参数 -> 睡眠 -> 返回 x+1
        print(x)          # 打印当前任务的参数（观察并发顺序）
        time.sleep(x / 5) # 模拟耗时操作（随 x 增加而增加）
        return x + 1      # 返回计算结果

    now = time.time()     # 记录串行执行的开始时间
    for i in range(10):   # 串行依次执行 10 次任务
        task(i)
    print("Without multi-processing:", time.time() - now)  # 打印串行总耗时

    now = time.time()     # 记录并行执行的开始时间
    runner = TaskRunner(task=task,        # 创建 TaskRunner，指定任务函数
                        arg_list=range(10),  # 参数列表为 0..9
                        max_workers=5)    # 最多并发 5 个进程
    runner.run()          # 启动并行运行
    print("Results:", runner.results_)  # 打印所有任务的 (status, arg, result) 列表
    print("Errors:", runner.errors_)    # 打印所有出错任务的参数列表
    print("With multi-processing:", time.time() - now)  # 打印并行总耗时
