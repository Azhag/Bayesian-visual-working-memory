#!/usr/bin/python

"""Track and display progress, providing estimated completion time.

This module provides 2 classes to simply add progress display to any
application. Programs with more complex GUIs might still want to use it for the
estimates of time remaining it provides.

Use the ProgressDisplay class if your work is done in a simple loop:
from progress import *
for i in ProgressDisplay(range(500)):
    do_work()

If do_work() doesn't send any output to stdout you can use the following,
which will cause a single status line to be printed and updated:
from progress import *
for i in ProgressDisplay(range(500), display=SINGLE_LINE):
    do_work()

For more complex applications, you will probably need to manage a Progress
object yourself:
from progress import *
progress = Progress(task_size)
for part in task:
    do_work()
    progress.increment()
    progress.print_status_line()

If you have a more sophisticated GUI going on, you can still use Progress
objects to give you a good estimate of time remaining:
from progress import *
progress = Progress(task_size)
for part in task:
    do_work()
    progress.increment()
    update_gui(progress.percentage(), progress.time_remaining())

This code is released under the Python 2.6.2 license.
"""

__author__ = ("Tim Newsome <nuisance@casualhacker.net>")
__version__ = "1.0.0"

import time
import math
import sys

# When _gather_stats is true, every time a Progress class is used to completion
# it will log statistics to ~/.progress_stats
_gather_stats = False
if _gather_stats:
    import os
    import os.path
    _stats_file = os.path.join(os.environ["HOME"], ".progress_stats")

MULTI_LINE = 0
SINGLE_LINE = 1

def _time():
    """Return time in seconds. I made a separate function so I can easily
    simulate an OS where that number is only accurate to the nearest second.
    """
    #return int(time.time())
    return time.time()

def time_string(seconds):
    """Return a human-friendly string representing a time duration.

    Examples:
    >>> time_string(12345)
    '3h26m'

    >>> time_string(61)
    '1m1s'

    >>> time_string(0)
    '0s'
    """
    if seconds < 0:
        return "--"
    units = [(60*60*24, "d"), (60*60, "h"), (60, "m"), (1, "s")]
    parts = []
    for i in range(len(units)):
        unit = units[i]
        if seconds >= unit[0]:
            break

    if i < len(units) - 1:
        # Round to the smallest unit that will be displayed.
        seconds = units[i+1][0] * round(float(seconds) / units[i+1][0])
        n = int(seconds / unit[0])
        parts.append("%d%s" % (n, unit[1]))
        seconds -= n * unit[0]
        unit = units[i+1]
    n = round(seconds / unit[0])
    parts.append("%d%s" % (n, unit[1]))

    return "".join(parts)

def quantity_string(quantity, unit, computer_prefix=False):
    """Return a human-friendly string representing a quantity by adding
    prefixes and keeping the number of significant figures low.
    'computer_prefix' determines whether each prefix step represents 1024
    or 1000.

    Examples:
    >>> quantity_string(1024, "B", True)
    '1.0KB'

    >>> quantity_string(40000, "m", False)
    '40km'

    >>> quantity_string(0.01, "m", False)
    '0.010m'
    """
    if quantity == 0:
        return "0%s" % unit
    # Units like m, B, and Hz are typically written right after the number.
    # But if your unit is "file" or "image" then you'll want a space between
    # the number and the unit.
    if len(unit) > 2:
        space =  " "
    else:
        space = ""
    if computer_prefix:
        prefixes = ["", "K", "M", "G", "T"]
        prefix_multiplier = 1024
    else:
        prefixes = ["", "k", "M", "G", "T"]
        prefix_multiplier = 1000
    divisor = 1
    for p in prefixes:
        digits = int(math.log10(quantity / divisor)) + 1
        if digits <= 3:
            format = "%%.%df%s%s%s" % (max(2 - digits, 0), space, p, unit)
            return format % (float(quantity) / divisor)
        divisor *= prefix_multiplier

    # No prefix available. Go scientific.
    return "%.2e%s%s"% (quantity, space, unit)

def rate_string(rate, work_unit, computer_prefix=False):
    """Return a human-friendly string representing a rate.  'rate' is given
    in 'work_unit's per second. If the rate is less than 0.1 then the inverse
    is shown.
       
    Examples:
    >>> rate_string(200000, "B", True)
    '195KB/s'

    >>> rate_string(0.01, "file")
    '1m40s/file'

    >>> rate_string(1.0 / 24 / 3600, "earthrot")
    '1d0h/earthrot'
    """
    if rate > 0 and rate < 0.1:
        return "%s/%s" % (time_string(1.0 / rate), work_unit)
    else:
        return "%s/s" % (quantity_string(rate, work_unit, computer_prefix))

class Progress:
    """Contains all state for a progress tracker."""
    def __init__(self, total_work, unit=None, computer_prefix=None):
        """Create a new progress tracker.
        'total_work' is the units of work that will be done.
        'unit' is the unit to be displayed to the user.
        'computer_prefix' should be set to True if this unit requires prefix
        increments of 1024 instead of the traditional 1000. If it is not set,
        then the class tries to guess based on 'unit'.
        """
        self.total_work = total_work
        self.unit = unit
        if computer_prefix is None and not self.unit is None:
            self.computer_prefix = unit.lower() in ["b", "bit", "byte"]
        else:
            self.computer_prefix = computer_prefix
        self.history = []
        if _gather_stats:
            self.stats_written = False
            self.log = []
            self.predicters = [self.predicted_rate, self._predicted_rate_period,
                    self._predicted_rate_avg, self._predicted_rate_pessimist]
        self.update(0)
        # Store away the beginning time so we can report the overall work rate.
        self.start = self.history[0]

        # stats for predicted_rate_pessimist
        self.pes_squares = 0
        self.pes_total = 0
        self.pes_samples = 0

    def update(self, work):
        """Updates the work completed to 'work'."""
        if work > self.total_work:
            self.total_work = work
        t = _time()

        history_entry = (work, t)

        # Only add history elements if the time is different from the previous
        # time, and at least a second has elapsed.
        replace = (len(self.history) > 0 and t == self.history[-1][1]) or \
            (len(self.history) > 1 and
             self.history[-1][1] - self.history[-2][1] < 1)

        # Keep track of sum of squared(time) per unit of work.
        # This has to happen "atomically" with adding the elements to history.
        if not replace and len(self.history) > 1:
            # Base computation on the last 2 history entries instead of
            # (work, t) because the new entry will likely be replaced later.
            delta_t = float(self.history[-1][1] - self.history[-2][1])
            delta_w = self.history[-1][0] - self.history[-2][0]
            rate = delta_t / delta_w
            self.pes_squares += rate * rate
            self.pes_total += rate
            self.pes_samples += 1

        if replace:
            self.history[-1] = history_entry
        else:
            self.history.append(history_entry)

        if _gather_stats:
            log_entry = (work, t, map(apply, self.predicters))
            if replace:
                self.log[-1] = log_entry
            else:
                self.log.append(log_entry)

        if _gather_stats and work >= self.total_work and not self.stats_written:
            # Work is complete. Write stats to a file.
            # import os.path
            stats_file = open(_stats_file, "a")
            stats_file.write(" ".join(sys.argv))
            stats_file.write(":")
            for avg, stddev in self._grade_performance():
                stats_file.write(" %.1f +- %.1f" % (avg, stddev))
            stats_file.write("\n")
            stats_file.close()
            self.stats_written = True

    def increment(self):
        """Increments the work completed by 1 unit."""
        self.update(self.history[-1][0] + 1)

    def percentage(self):
        """Returns the percent of work completed so far."""
        return 100.0 * self.history[-1][0] / self.total_work

    def done(self):
        """Returns True when all work is done."""
        return self.history[-1][0] == self.total_work

    def _predicted_rate_period(self):
        """Returns the predicted rate of work in units per second for the
        remainder of the work. Assumes that next n minutes will be like the
        last n minutes. In other words, if only 10% of work is remaining,
        only look to see how long it took to complete the last 10% of the
        work.
        """
        if len(self.history) < 2:
            return None
        work_done = self.history[-1][0]
        remaining_work = self.total_work - work_done
        # Drop all old history entries.
        while work_done - self.history[1][0] > remaining_work:
            self.history.pop(0)
        return float(self.history[-1][0] - self.history[0][0]) / \
                (self.history[-1][1] - self.history[0][1])

    def _predicted_rate_avg(self):
        """Returns the predicted rate of work in units per second for the
        remainder of the work. Assumes that next n minutes will be like the
        average has been so far.
        """
        if len(self.history) < 2:
            return None
        # work_done = self.history[-1][0]
        return float(self.history[-1][0] - self.start[0]) / \
                (self.history[-1][1] - self.start[1])

    def _predicted_rate_pessimist(self):
        """Returns the predicted rate of work in units per second for the
        remainder of the work. Assumes each remaining unit will take the time
        it takes to process 1 unit plus 1 standard deviation. Scale this
        pessimism by the percentage of work complete. This function is very
        unlikely to overestimate the work rate when the work is almost done.
        """
        if len(self.history) < 3:
            return self._predicted_rate_avg()
        avg = self.pes_total / self.pes_samples
        stddev = math.sqrt(self.pes_squares / self.pes_samples - avg * avg)
        return 1.0 / (avg + stddev * self.percentage() / 100)

    def predicted_rate(self):
        """Returns the predicted rate of work in units per second for the
        remainder of the work.
        """
        rate_1 = self._predicted_rate_period()
        if rate_1 is None:
            return None
        rate_3 = self._predicted_rate_pessimist()
        if rate_3 is None:
            return rate_1
        return (rate_1 + rate_3) / 2

    def predicted_rate_str(self):
        """Returns self.predicted_rate() as a string."""
        return rate_string(self.predicted_rate(), self.unit,
                self.computer_prefix)

    def overall_rate(self):
        """Returns the overall rate of work so far in units per second."""
        if self.time_elapsed() == 0:
            return 1
        return float(self.history[-1][0] - self.start[0]) / self.time_elapsed()

    def overall_rate_str(self):
        """Returns self.overall_rate() as a string."""
        if self.unit is None:
            unit = "unit"
        else:
            unit = self.unit
        return rate_string(self.overall_rate(), unit, self.computer_prefix)

    def time_elapsed(self):
        return self.history[-1][1] - self.start[1]

    def time_remaining(self):
        """Returns the estimated amount of time (in seconds) remaining until
        all the work is complete."""
        work_rate = self.predicted_rate()
        if work_rate is None:
            return -1
        remaining_work = self.total_work - self.history[-1][0]
        work_time_remaining = remaining_work / work_rate
        work_time_elapsed = _time() - self.history[-1][1]
        return work_time_remaining - work_time_elapsed

    def time_remaining_str(self):
        """Returns self.time_remaining() as a string."""
        return time_string(self.time_remaining())

    def eta(self):
        """Returns the time (similar to time.time()) when the work will be
        complete."""
        return _time() + self.time_remaining()

    def eta_str(self, format="%d.%m %H:%M"):
        """
            Returns a formatted time description of the time of job completion
        """
        return time.strftime(format, time.localtime(self.eta()))

    def status_line(self, task=None):
        """Return a status line. Optionally a task name can be passed in to be
        included as well."""
        status = str(self)
        if task is None:
            line = status
        else:
            status += " | "
            line = "%s%s" % (status, task)
        return line

    def print_status_line(self, task=None, line_length=78):
        """Write a status line to stdout. If the work is completed then the
        cursor is moved to the next line, but otherwise it is kept on the
        current one. Optionally a task name can be passed in to be displayed
        as well."""
        if self.done():
            eol = "\n"
        else:
            eol = "\r"
        line = self.status_line(task)
        line = line[:line_length]
        sys.stdout.write("%s%s%s" % (line, " " * (78-len(line)), eol))
        sys.stdout.flush()

    def __str__(self):
        """Returns the overall progress as a user-presentable string."""
        parts = []
        parts.append("[%d%%]" % int(self.percentage()))
        if not self.unit is None and self.time_elapsed() > 0:
            parts.append(self.overall_rate_str())
        if self.done():
            parts.append(time_string(self.time_elapsed()))
        else:
            if _gather_stats:
                remaining_work = self.total_work - self.history[-1][0]
                for p in self.predicters:
                    predicted = p()
                    if predicted is None:
                        continue
                    work_time_remaining = remaining_work / predicted
                    work_time_elapsed = _time() - self.history[-1][1]
                    parts.append(time_string(
                                work_time_remaining - work_time_elapsed))
            else:
                parts.append(self.time_remaining_str())
        return " ".join(parts)

    def _grade_performance(self):
        """Go through the internal log to see how well time remaining was
        predicted. Returns average and standard deviation for the predicted
        time elapsed divided by the actual time elapsed."""
        end = self.log[-1]
        entry_count = 0
        algorithms = len(end[2])
        total = [0] * algorithms
        squares = [0] * algorithms
        average = [0] * algorithms
        stddev = [0] * algorithms
        # Ignore the first entry, since no prediction can be made based on
        # just one entry.
        for entry in self.log[1:-1]:
            for i in range(algorithms):
                predicted = entry[1] + (end[0] - entry[0]) / entry[2][i] - \
                            self.start[1]
                actual = end[1] - self.start[1]
                factor_percent = 100.0 * predicted / actual
                total[i] += factor_percent
                squares[i] += factor_percent * factor_percent
            entry_count += 1
        if entry_count == 0:
            return []
        for i in range(algorithms):
            average[i] = total[i] / entry_count
            stddev[i] = math.sqrt(squares[i] / entry_count - \
                    average[i]*average[i])
        return zip(average, stddev)

class ProgressDisplay:
    """Wraps an iterator and displays progress every time next() is called.
    In order to show progress, it computes the total size of the data by
    expanding the iterator into a list. With non-trivial iterators this can
    take a long time, and will take a lot of memory if the expansion is large.
    The class is intended to keep simple programs simple. In more complex
    applications it's better to manage your own Progress class.
    """
    def __init__(self, iterator, unit=None, computer_prefix=None, display=MULTI_LINE):
        """Create a new progress display.
        'iterator' is the iterator containing the work to be done.
        'unit' is the unit to be displayed to the user.
        'computer_prefix' should be set to True if this unit requires prefix
        increments of 1024 instead of the traditional 1000. If it is not set,
        then the class tries to guess based on 'unit'.
        'display' defaults to MULTI_LINE to print a new line for every update,
        or can be SINGLE_LINE to keep updating a single status line.
        """
        if hasattr(iterator, "__len__"):
            # This may be an expensive operation, for instance on a
            # hypothetical os.walk() which implements __len__.
            length = len(iterator)
            self.iterator = iter(iterator)
        else:
            list = []
            # TODO: isn't there some kind of builtin expand operation?
            for i in iterator:
                list.append(i)
            length = len(list)
            self.iterator = iter(list)
        self.progress = Progress(length, unit, computer_prefix)
        self.display = display
        # The first call to next is before the work actually starts, so we
        # shouldn't increment() at that point.
        self.first = True

    def __iter__(self):
        return self

    def next(self):
        if self.first:
            self.first = False
        else:
            self.progress.increment()
        if self.display == SINGLE_LINE:
            self.progress.print_status_line()
        else:
            print self.progress.status_line()
        return self.iterator.next()

def _test():
    import doctest
    return doctest.testmod()

def _demo_sleep():
    for i in ProgressDisplay(range(5)):
        time.sleep(1)

def _demo_walk():
    import os.path
    size = 0
    top_dir = "/lib"
    for root, dirs, files in \
            ProgressDisplay(os.walk(top_dir), "dir", display=SINGLE_LINE):
        for f in files:
            file = os.path.join(root, f)
            if os.path.exists(file):
                size += os.path.getsize(file)
    print size, "bytes in", top_dir

def _demo_file():
    import os
    import subprocess

    def identify(file):
        output = subprocess.Popen(["file", "-b", "-L", file],
                stdout=subprocess.PIPE).communicate()[0]
        output = output.strip()
        return output

    top_dir = "/etc"

    # Figure out how many files need to be identified.
    file_count = 0
    for root, dirs, files in os.walk(top_dir):
        file_count += len(files)

    # Identify the files.
    types = {}
    progress = Progress(file_count, "file")
    for root, dirs, files in os.walk(top_dir):
        for file in files:
            target = os.path.join(root, file)
            progress.print_status_line(target)
            type = identify(target)
            if type in types:
                types[type] += 1
            else:
                types[type] = 1
            progress.increment()

    progress.print_status_line()

    # Print a report.
    results = []
    for type in types:
        results.append((types[type], type))
    results.sort(reverse=True)
    print "top 10 file types in %s:" % top_dir
    for result in results[:10]:
        print "%d: %s" % (result)

if __name__ == "__main__":
    failures = _test()[0]
    if failures == 0:
        _demo_sleep()
        _demo_walk()
        _demo_file()
