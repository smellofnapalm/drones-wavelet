# Лаба про анализ сигнала с дрона
# Чирков Михаил, январь 2024
# База записей с радара на основе эффекта Доплера выложена в диске.
# Пацанам: Попробовать отделить какие-то два класса (достаточно рассмотреть по 2-3 файла из каждого класса), можно выбрать любые классы, но самое разумное отделить дронов(больших и маленьких) от всех других классов
import os

import numpy as np
import matplotlib.pyplot as plt
import pywt
import scaleogram as scg
from scipy.io import wavfile
from skimage.feature import peak_local_max
from dataclasses import dataclass
import pickle
import queue
import copy


# FUNCTIONS

@dataclass
class ClusterInfo:
    cluster: list[int]
    k: float
    b: float
    center: (float, float)
    mean_amp: float
    sigma_amp: float
    sigma_period: float
    size: int


@dataclass
class PeaksInfo:
    peaks_info: list[(float, float, float)]
    samplerate: float
    cwt: scg.CWT
    mean_amp: float
    sigma_amp: float
    clusters: list[ClusterInfo]


# Прочитать данные из .wav файла
def read_wav(filename):
    samplerate, data = wavfile.read(filename)
    if len(data.shape) >= 2:
        data = np.mean(data, axis=1)
    return data, samplerate


# Основная функция -- подсчет PeaksInfo по сигналу и набору параметров
def find_maximal(signal, samplerate, periods_limits, min_distance, threshold_abs_sigma,
                 num_peaks, radius2, alpha) -> PeaksInfo:
    scales = scg.periods2scales(np.logspace(np.log10(periods_limits[0]), np.log10(periods_limits[1]), num=100),
                                scg.get_default_wavelet())
    # time = np.arange(0, len(signal) / samplerate, step=1.0 / samplerate)
    time = np.arange(0, len(signal))
    cwt = scg.CWT(time, signal, scales)
    if alpha is None:
        alpha = np.sqrt(len(signal) / (cwt.scales[-1] - cwt.scales[0]))
    xydata = np.abs(cwt.coefs)
    mean = np.mean(xydata)
    sigma = np.std(xydata)
    threshold_abs = mean + threshold_abs_sigma * sigma
    xy = peak_local_max(xydata, min_distance=min_distance, threshold_abs=threshold_abs, num_peaks=num_peaks)
    peak_mask = np.zeros_like(cwt.coefs, dtype=bool)
    peak_mask[tuple(xy.T)] = True
    maximal_values = xydata[peak_mask == True]
    maximal_periods = [1.0 / pywt.scale2frequency(scg.get_default_wavelet(), cwt.scales[int(xy[i, 0])]) for i in
                       range(len(xy))]
    maximal_times = [cwt.time[int(xy[i, 1])] for i in range(len(xy))]
    peaks_info = list(zip(maximal_times, maximal_periods, maximal_values))
    closest = all_closest(peaks_info, radius2, alpha)
    clusters = clusterize(closest)
    clusters_info = []
    for cluster in clusters:
        xs = np.array([peaks_info[cluster[i]][0] for i in range(len(cluster))])
        ys = np.array([peaks_info[cluster[i]][1] for i in range(len(cluster))])
        amps = np.array([peaks_info[cluster[i]][2] for i in range(len(cluster))])
        mean_amp = np.mean(amps)
        sigma_amp = np.std(amps)
        center = (np.mean(xs), np.mean(ys))
        k, b = linear_regression(xs, ys)
        sigma_period = np.std(ys)
        clusters_info.append(ClusterInfo(cluster, k, b, center, mean_amp, sigma_amp, sigma_period, len(cluster)))
    clusters_info.sort(key=lambda el: -el.size)
    return PeaksInfo(peaks_info, samplerate, cwt, mean, sigma, clusters_info)


# Метрика на скалограмме
def distance2(x1, y1, x2, y2, alpha):
    return np.abs(x1 - x2) + alpha * np.abs(y1 - y2)


# Поиск точек, находящихся на расстоянии <= radius2 по метрике distance2
def find_closest(peaks_info, i, radius2, alpha):
    x1, y1 = peaks_info[i][0], peaks_info[i][1]
    return [j for j in range(len(peaks_info)) if
            distance2(x1, y1, peaks_info[j][0], peaks_info[j][1], alpha) <= radius2]


# Для каждой точки найти ее соседей
def all_closest(peaks_info, radius2, alpha):
    return [find_closest(peaks_info, i, radius2, alpha) for i in range(len(peaks_info))]


# Кластеризация при помощи поиска в ширину
def clusterize(closest):
    used = [False for _ in range(len(closest))]
    sets = []
    for i in range(len(closest)):
        if used[i]:
            continue
        q = queue.Queue()
        s = [i]
        q.put(i)
        used[i] = True
        while not q.empty():
            i = q.get()
            for j in closest[i]:
                if not used[j]:
                    s.append(j)
                    q.put(j)
                    used[j] = True
        sets.append(copy.copy(s))
    return sets


# Приближение кластера прямой (отрезком на картинке)
def linear_regression(xs, ys):
    matrix = np.vstack([xs, np.ones(len(xs))]).T
    k, b = np.linalg.lstsq(matrix, ys, rcond=None)[0]
    return k, b


# Отрисовка кластера (inds) на заданом ax
def draw_cluster(ax, peaks_info, inds):
    xs = np.array([peaks_info[i][0] for i in inds])
    ys = np.array([peaks_info[i][1] for i in inds])
    color_list = np.random.rand(3)
    ax.scatter(xs, ys, color=(color_list[0], color_list[1], color_list[2]))
    k, b = linear_regression(xs, ys)
    border = np.array([np.min(xs), np.max(xs)])
    ax.plot(border, k * border + b, 'r')


# Загрузка данных с диска или подсчет, если calculate=True
def load_info(filename, picklename, periods_limits=(50, 5000), min_distance=10, threshold_abs_sigma=2.0, calculate=True,
              num_peaks=np.inf, radius2=1000.0, alpha=None):
    signal, samplerate = read_wav(filename)
    if calculate:
        info = find_maximal(signal, samplerate, periods_limits, min_distance, threshold_abs_sigma, num_peaks, radius2,
                            alpha)
        pickle.dump(info, open(picklename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    info = pickle.load(open(picklename, 'rb'))
    return info


# Вычленить имена для .wav файлов и .pkl файлов из папки
def names_from_folder(folder):
    wavs = list(map(lambda el: folder + '/' + el, os.listdir(folder)))
    pickles = list(map(lambda el: folder + '_info/' + el.split('.')[0] + '.pkl', os.listdir(folder)))
    return wavs, pickles


# Общая функция отрисовки
def draw_all_clusters(info: PeaksInfo):
    ax = scg.cws(info.cwt, coikw=coikw, figsize=(12, 6), yscale='log')
    for cluster_info in info.clusters:
        draw_cluster(ax, info.peaks_info, cluster_info.cluster)
    plt.show()


# Общая функция по выводу числовых данных (фантазия не ограничена тем, что тут представленно, можно сделать куда больше...)
def print_info(name, info: PeaksInfo):
    # Выбирается топовый по размеру кластер
    top_cluster = info.clusters[0]
    separator = '================================\n'

    print(f'Данные файла {name}')
    print(separator)
    print('Информация про самый крупный (главный кластер) точек максимума:')
    print(f'    Размер главного кластера = {top_cluster.size}, {int(top_cluster.size / len(info.peaks_info) * 100)} % от всех точек максимума')
    print(f'    Средний период главного кластера = {top_cluster.center[1]}')
    print(f'    Стандартное отклонение периода главного кластера = {top_cluster.sigma_period / top_cluster.center[1]}')
    print(f'    Средняя амплитуда главного кластера = {top_cluster.mean_amp}')
    print(f'    Стандартное отклонение амплитуды главного кластера = {top_cluster.sigma_amp / top_cluster.mean_amp}')
    print(f'    Угловой коэффициент = {np.arctan(top_cluster.k) / np.pi * 180.0}\n')
    print(separator)

    print('Информация про все данные целиком')
    print(f'    Средняя амплитуда = {info.mean_amp}')
    print(f'    Стандартное отклонение амплитуды = {info.sigma_amp}')
    print(f'    Средневзвешенный период точек максимума = {np.average(np.array([el[1] for el in info.peaks_info]), weights=np.array([el[2] for el in info.peaks_info]))}')
    print(f'    Всего кластеров на секунду времени  = {len(info.clusters) / (len(info.cwt.signal) / info.samplerate)}')
    print(f'    Средняя длина кластера = {np.mean(np.array([cluster.size for cluster in info.clusters]))}')
    print(f'    Частота дискретизации = {info.samplerate} Hz\n')
    print(separator)


# INITIAL

# Выбор вейвлета
coikw = {'alpha': 0.5, 'hatch': '/'}
base_wavelet = 'cmor1.0-1.5'
scg.set_default_wavelet(base_wavelet)
samplerate = 96000

# Запоминание имен файлов для анализа
big_drones = names_from_folder('drone_db/small_db/big_drone')
bird = names_from_folder('drone_db/small_db/bird')
free_space = names_from_folder('drone_db/small_db/free space')
people = names_from_folder('drone_db/small_db/people')
small_copter = names_from_folder('drone_db/small_db/small_copter')

# Подсчет данных по файлам (или загрузка с диска, если calculate = False) 
info_big_drone1 = load_info(big_drones[0][0], big_drones[1][0], periods_limits=(50, 5000), min_distance=10,
                            calculate=False)
info_big_drone2 = load_info(big_drones[0][6], big_drones[1][6], periods_limits=(500, 1300), radius2=1000.0,
                            min_distance=10, calculate=False)
info_bird = load_info(bird[0][0], bird[1][0], periods_limits=(50, 5000), min_distance=10, calculate=False)
info_small_copter = load_info(small_copter[0][0], small_copter[1][0], periods_limits=(50, 5000), min_distance=10,
                              calculate=False)
info_free_space = load_info(free_space[0][0], free_space[1][0], periods_limits=(50, 5000), min_distance=10,
                            calculate=False)
info_people = load_info(people[0][0], people[1][0], periods_limits=(50, 5000), min_distance=10, calculate=False)

# Напечатать некоторую числовую информацию о данных
print_info(big_drones[0][0], info_big_drone1)
print_info(big_drones[0][6], info_big_drone2)
print_info(bird[0][0], info_bird)
print_info(small_copter[0][0], info_small_copter)
print_info(free_space[0][0], info_free_space)
print_info(people[0][0], info_people)

# Отрисовка вейвлет-преобразований вместе с кластерами
draw_all_clusters(info_big_drone1)
draw_all_clusters(info_people)

plt.show()