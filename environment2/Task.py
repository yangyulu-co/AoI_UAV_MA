
class Task:
    """任务"""
    def __init__(self, storage: float=5*(10**4), compute: float=5*(10**4)*200):
        self.storage = storage
        """任务的大小，bit数"""
        self.compute = compute
        """计算任务需要的CPU周期"""




    def get_storage_require(self)->float:
        """返回任务所需的存储空间的大小"""
        return self.storage
    def get_compute_require(self)->float:
        """返回任务所需的计算能力"""
        return self.compute
    def get_task_require(self)->[float]:
        """返回任务所需的存储和计算能力需求"""
        return [self.storage,self.compute]