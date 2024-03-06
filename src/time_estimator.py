import time
import numpy as np
import pandas as pd
import random

from graph import plot_estimated_times
from typing import Callable
import threading


class Simulation:
    def __init__(self, estimated_time):
        self.estimated_time = estimated_time

    def save(self):
        print(f"Simulation saved with estimated time: {self.estimated_time}")


def get_callback(simulation):
    def db_save(estimated_time):
        print(
            f"********************************* Estimated time: {simulation.estimated_time}"
        )
        simulation.estimated_time = round(estimated_time, 2)
        simulation.save()
        print("********************************* Simulation updated")

    def update_simulation_status(estimated_time):
        thread = threading.Thread(target=db_save, args=(estimated_time,))
        thread.start()

    return update_simulation_status


class TimeEstimator:
    def __init__(
        self,
        iter_range: range,
        span: int = None,
        callback: Callable[[float], None] = None,
        recent_ema_weight: float = 0.6,
    ):
        self.iter_range = iter_range
        self.total_iterations = self._calculate_total_iterations()

        if callback is None:
            callback = self.default_callback

        if span is None:
            span = self._calculate_optimal_span()
        elif span < 2:
            raise ValueError(
                "Span must be at least 2. If you did not specify a span, try a larger range."
            )

        self.times = np.zeros(span)
        self.last_median = 0
        self.current = 0

        self.span = span
        self.callback = callback
        self.recent_ema_weight = recent_ema_weight
    
    def _calculate_total_iterations(self):
        start = self.iter_range.start
        stop = self.iter_range.stop
        step = self.iter_range.step
        total_iterations = (stop - start) // step
        return total_iterations

    def _calculate_optimal_span(self):
        return self.total_iterations // 20

    def push(self, time: float, current_iteration: int = 0):
        self.times[self.current] = time
        self.current = self.current + 1

        if self.current == self.span:
            self.current = 0
            self.callback(self.estimate_time(current_iteration))
            self.times[:] = 0

    # def _moving_average(self) -> float:
    #     mean_time = np.mean(self.durations)
    #     return mean_time

    def _iqr_filter(self, diffs: pd.Series):
        q1 = diffs.quantile(0.10)
        q3 = diffs.quantile(0.60)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        filtered_durations = diffs[(diffs >= lower_bound) & (diffs <= upper_bound)]

        # formatted_s = filtered_durations.map('{:.2f}'.format)
        # formatted_d = diffs.map('{:.2f}'.format)
        # print(f'Filtered: {formatted_s.to_list()}, \nOriginal: {formatted_d.to_list()}')

        return filtered_durations

    def _ema(self, diffs):
        filtered = self._iqr_filter(diffs)
        ema = filtered.ewm(span=self.span).mean()
        return ema.median()

    def estimate_time(self, current_iteration):
        diffs = np.diff(self.times)
        diffs[diffs < 0] = 0
        diffs = pd.Series(diffs)

        new_ema = self._ema(diffs)

        if self.last_median <= 0:
            calc_duration = new_ema
        else:
            calc_duration = new_ema * self.recent_ema_weight + self.last_median * (
                1 - self.recent_ema_weight
            )

        self.last_median = calc_duration

        start = self.iter_range.start
        step = self.iter_range.step

        current_iteration = (current_iteration - start) // step

        iterations_left = self.total_iterations - current_iteration
        time_left = iterations_left * calc_duration
        return time_left

    @staticmethod
    def default_callback(time):
        with open("time_estimator.log", "a") as file:
            file.write(f"{time}\n")
        print(
            f"=================================================== ------ Estimated time left: {time}"
        )


if __name__ == "__main__":

    def test_time_estimator():
        est_list = []

        def callback(time):
            print(f"Estimated time left: {time}")
            est_list.append(time)

        with open("./diffs.txt", "r") as file:
            execution_times = [float(line.strip()) for line in file]

        total_iterations = range(0, len(execution_times))
        # total_iterations = range(0, 1000, 3)

        time_estimator = TimeEstimator(
            total_iterations, span=30, callback=callback)

        start = time.time()
        total = time.time()
        time_estimator.push(total)

        for i in total_iterations:
            total += execution_times[i]
            time_estimator.push(total, i)

        # for i in total_iterations:
        #     random_time = random.randint(10, 15) / 100
        #     time.sleep(random_time)
        #     time_estimator.push(time.time(), i)

        end = time.time()
        print(f"Total time: {end - start}")

        plot_estimated_times(data=est_list)

    test_time_estimator()
    # test()
