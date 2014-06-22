#!/usr/bin/env python
# encoding: utf-8
"""
keyboardthreadtest.py

Created by Loic Matthey on 2014-04-19
Copyright (c) 2014 . All rights reserved.
"""

import thread
import time
import collections
import threading


def thread_input_run(input_queue):
    while True:
        input_char = raw_input("What to do? ")

        if input_char:
            input_queue.append(input_char)

## Test multithread implementation of keyboard interaction
def main_oneway():
    '''
        Instantiate a thread for the keyboard, see if it can print stuff
    '''

    input_queue = collections.deque()

    thread.start_new_thread(thread_input_run, (input_queue,) )

    while True:
        # print "waiting for input..."
        time.sleep(2.0)

        while input_queue:
            # Pop it
            elem_from_queue = input_queue.popleft()

            print "\n!New element!!", elem_from_queue


class BigComputation():
    def __init__(self, val=0):
        self.val = val

    def run(self):
        while True:
            print "computing..."
            time.sleep(3.0)

            self.val += 1

class ComputationKeyboardThread(threading.Thread):
    def __init__(self, big_computation):
        threading.Thread.__init__(self)

        self.daemon = True
        self.big_computation = big_computation

    def run(self):
        while True:
            input_char = raw_input()

            if input_char == 'p':
                print "\nBigComputation value: ", self.big_computation.val, "\n"
            else:
                print "p:   print computation value"
                print "h:   print this help message"

def main_secondway():

    # Get a computation
    big_comput = BigComputation()

    # Get a thread to handle keyboard
    keyboard_thread = ComputationKeyboardThread(big_comput)
    keyboard_thread.start()

    # Start the computation
    big_comput.run()



if __name__ == '__main__':
    # main_oneway()
    main_secondway()
