# 作为暂时测试用
from environment2.Area import generate_solution
from environment2.Constant import ETUAV_height
from environment2.DPUAV import DPUAV
from environment2.ETUAV import ETUAV
from environment2.Position import Position
from environment2.Task import Task
from environment2.UE import UE
import matplotlib.pyplot as plt
if __name__=='__main__':
    # ue = UE(Position(0,0,0))
    # ue.task = Task(5000,5)
    # dpuav = DPUAV(Position(30,0,40))
    # aoi_1 = dpuav.calcul_single_compute_and_offloading_aoi(ue,1)
    # aoi_2 = dpuav.calcul_single_compute_and_offloading_aoi(ue,2)
    # print(aoi_1,aoi_2)
    # energy_1 = dpuav.calcul_single_compute_and_offloading_energy(ue,1)
    # energy_2 = dpuav.calcul_single_compute_and_offloading_energy(ue,2)
    # print(energy_1,energy_2)

    # print(generate_solution(1))

    uee = UE(Position(0,0,0))
    uee.discharge(uee.get_energy())
    print(uee.get_energy())
    x = [i for i in range(500)]
    y = [0 for i in range(500)]
    for i in range(500):
        etuav = ETUAV(Position(i, 0, ETUAV_height))
        y[i] = etuav.get_charge_energy(uee)
    plt.plot(x,y)
    plt.plot(x,[5*(10**(-7))]*500)
    plt.show()
