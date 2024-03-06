import time
import numpy as np
import pandas as pd
import logging

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


logger = logging.getLogger("TIME_ESTIMATOR")
logger.setLevel(logging.INFO)
SPAN_THRESHOLD = 2


def deactivate_below_threshold(threshold):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if getattr(self, "span") < threshold:
                return None
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


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

        self.span = span
        if span < SPAN_THRESHOLD:
            logger.warning(
                f" The iteration range is too small or you set a span less than 2. Time estimation will be deactivated.\
                    \nMake sure the loop executes at least {SPAN_THRESHOLD} iterations if you want time estimation."
            )
            return
        if span > self.total_iterations:
            logger.warning(
                " The span is too large. It should be less than the total number of iterations."
            )
            return

        self.times = np.zeros(span)
        self.last_median = 0
        self.current = 0

        self.callback = callback
        self.recent_ema_weight = recent_ema_weight

    def _calculate_total_iterations(self):
        start = self.iter_range.start
        stop = self.iter_range.stop
        step = self.iter_range.step
        total_iterations = (stop - start) // step
        return total_iterations

    def _calculate_optimal_span(self):
        if self.total_iterations >= 100:
            span = self.total_iterations // 20
            message = f" Span for time estimation is: {span}. You will see the estimation every {span} iterations."
        else:
            if self.total_iterations < 100 and self.total_iterations >= 20:
                span = 5
            elif self.total_iterations < 20 and self.total_iterations >= 3:
                span = 3
            elif self.total_iterations == 2:
                span = 2
            else:
                span = -1
            message = f" Span for time estimation is: {span}. You will see the estimation every {span} iterations."
            message += "\nThe estimation will not be very accurate with a span less than 10. Consider increasing the number of iterations."

        logger.info(message)
        return span

    @deactivate_below_threshold(SPAN_THRESHOLD)
    def push(self, time: float, current_iteration: int = 0):
        self.times[self.current] = time
        self.current = self.current + 1

        if self.current == self.span:
            self.current = 0
            self.callback(self.estimate_time(current_iteration))
            self.times[:] = 0

    def _iqr_filter(self, diffs: pd.Series):
        q1 = diffs.quantile(0.10)
        q3 = diffs.quantile(0.60)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        filtered_durations = diffs[(diffs >= lower_bound) & (diffs <= upper_bound)]

        return filtered_durations

    def _ema(self, diffs):
        filtered = self._iqr_filter(diffs)
        ema = filtered.ewm(span=self.span).mean()
        return ema.median()

    @deactivate_below_threshold(SPAN_THRESHOLD)
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

        time_estimator = TimeEstimator(total_iterations, callback=callback)

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

        # plot_estimated_times(data=est_list)

    test_time_estimator()
    # test()
