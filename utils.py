from os import listdir
from os.path import isfile, join, split
import threading
import json
import codecs
from queue import Queue
import requests
import multiprocessing
import logging


# list files in a folder and put them in to a queue for multi-threading processing
def multi_thread_process_files(dir_path, file_extension, num_threads, process_func,
                               proc_desc='processed', args=None, multi=None,
                               file_filter_func=None, callback_func=None,
                               thread_wise_objs=None):
    onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    num_pdfs = 0
    files = None if multi is None else []
    lst = []
    for f in onlyfiles:
        if f.endswith('.' + file_extension) if file_filter_func is None \
                else file_filter_func(f):
            if multi is None:
                lst.append(join(dir_path, f))
            else:
                files.append(join(dir_path, f))
                if len(files) >= multi:
                    lst.append(files)
                    files = []
            num_pdfs += 1
    if files is not None and len(files) > 0:
        lst.append(files)
    multi_thread_tasking(lst, num_threads, process_func, proc_desc, args, multi, file_filter_func,
                         callback_func,
                         thread_wise_objs=thread_wise_objs)


def multi_thread_tasking(lst, num_threads, process_func, args=None,  callback_func=None, thread_wise_objs=None,
                         thread_init_func=None, thread_end_func=None):
    """
    multithreadingly do a task over a list of objects
    Args:
        lst: the list of objects to process
        num_threads: number of threads
        process_func: the function to do the task
        args: a list of arguments to be passed to the function
        callback_func: a function to call back when finishes
        thread_wise_objs: an object to be shared by each single thread - very useful for situations like database connections
        thread_init_func: initialisation function when a thread starts
        thread_end_func: a function to be called when a thread ends (e.g. release resources like db connections)

    Returns:

    """
    num_tasks = len(lst)
    pdq_queue = Queue(num_tasks)
    # print('putting list into queue...')
    for item in lst:
        pdq_queue.put_nowait(item)
    thread_num = min(num_tasks, num_threads)
    arr = [process_func] if args is None else [process_func] + args
    arr.insert(0, pdq_queue)
    # print('queue filled, threading...')
    thread_objs = []
    for i in range(thread_num):
        tarr = arr[:]
        thread_obj = None
        if thread_wise_objs is not None and isinstance(thread_wise_objs, list):
            thread_obj = thread_wise_objs[i]
        if thread_obj is None and thread_init_func is not None:
            thread_obj = thread_init_func()
            thread_objs.append(thread_obj)
        tarr.insert(0, thread_obj)
        t = threading.Thread(target=multi_thread_do, args=tuple(tarr))
        t.daemon = True
        t.start()

    # print('waiting jobs to finish')
    pdq_queue.join()
    if thread_end_func is not None:
        for to in thread_objs:
            if to is not None:
                thread_end_func(to)
    # print('{0} files {1}'.format(num_pdfs, proc_desc))
    if callback_func is not None:
        callback_func(*tuple(args))


def multi_thread_tasking_it(it_lst, num_threads, process_func,
                            proc_desc='processed', args=None, multi=None,
                            file_filter_func=None, callback_func=None, thread_wise_objs=None):
    pdq_queue = Queue(1000)
    thread_num = num_threads
    arr = [process_func] if args is None else [process_func] + args
    arr.insert(0, pdq_queue)
    # print('queue filled, threading...')
    for i in range(thread_num):
        thread_arr = arr[:]
        thread_obj = None
        if thread_wise_objs is not None and isinstance(thread_wise_objs, list):
            thread_obj = thread_wise_objs[i]
        thread_arr.insert(0, thread_obj)
        t = threading.Thread(target=multi_thread_do, args=tuple(thread_arr))
        t.daemon = True
        t.start()

    # print('waiting jobs to finish')
    # print('putting list into queue...')
    for item in it_lst:
        pdq_queue.put(item)
    pdq_queue.join()
    # print('{0} files {1}'.format(num_pdfs, proc_desc))
    if callback_func is not None:
        callback_func(*tuple(args))


def multi_thread_do(thread_obj, q, func, *args):
    while True:
        p = q.get()
        try:
            if thread_obj is not None:
                func(thread_obj, p, *args)
            else:
                func(p, *args)
        except Exception as e:
            logging.error(u"error doing {0} on {1} \n{2}".format(func, p, str(e)))
        q.task_done()


def save_json_array(lst, file_path, encoding='utf-8'):
    with codecs.open(file_path, 'w', encoding=encoding) as wf:
        json.dump(lst, wf)


def save_string(str_content, file_path, encoding='utf-8'):
    with codecs.open(file_path, 'w', encoding=encoding) as wf:
        wf.write(str_content)


def load_json_data(file_path, encoding='utf-8'):
    data = None
    with codecs.open(file_path, encoding=encoding) as rf:
        data = json.load(rf, encoding=encoding)
    return data


def http_post_result(url, payload, headers=None, auth=None):
    req = requests.post(
        url, headers=headers,
        data=payload, auth=auth)
    return req.content


def http_get_content(url):
    req = requests.get(url)
    return req.content

def multi_thread_large_file_tasking(large_file, num_threads, process_func,
                                    proc_desc='processed', args=None, multi=None,
                                    file_filter_func=None, callback_func=None,
                                    thread_init_func=None, thread_end_func=None,
                                    file_encoding='utf-8'):
    num_queue_size = 1000
    pdq_queue = Queue.Queue(num_queue_size)
    print('queue filled, threading...')
    thread_objs = []
    for i in range(num_threads):
        arr = [process_func] if args is None else [process_func] + args
        to = None
        if thread_init_func is not None:
            to = thread_init_func()
            thread_objs.append(to)
        arr.insert(0, to)
        arr.insert(1, pdq_queue)
        t = threading.Thread(target=multi_thread_do, args=tuple(arr))
        t.daemon = True
        t.start()

    print('putting list into queue...')
    num_lines = 0
    with codecs.open(large_file, encoding=file_encoding) as lf:
        for line in lf:
            num_lines += 1
            pdq_queue.put(line)

    print('waiting jobs to finish')
    pdq_queue.join()
    if thread_end_func is not None:
        for to in thread_objs:
            if to is not None:
                thread_end_func(to)
    print('{0} lines {1}'.format(num_lines, proc_desc))
    if callback_func is not None:
        callback_func(*tuple(args))


def read_text_file(file_path, encoding='utf-8'):
    lines = []
    with codecs.open(file_path, encoding=encoding) as rf:
        lines += rf.readlines()
    return [l.strip() for l in lines]


def read_text_file_as_string(file_path, encoding='utf-8'):
    s = None
    with codecs.open(file_path, encoding=encoding) as rf:
        s = rf.read()
    return s


def multi_process_do(thread_obj, q, func, *args):
    while True:
        p = q.get()
        if p == 'EMPTY-NOW':
            q.task_done()
            break
        try:
            if thread_obj is not None:
                func(thread_obj, p, *args)
            else:
                func(p, *args)
        except Exception as e:
            logging.error(u'error doing {0} on {1} \n{2}'.format(func, p, str(e)))
        q.task_done()


def multi_process_tasking(lst, process_func, num_procs=multiprocessing.cpu_count()*2,
                          proc_desc='processed', args=None, multi=None,
                          file_filter_func=None, callback_func=None, thread_wise_objs=None,
                          thread_init_func=None, thread_end_func=None, thread_end_args=[]):
    """
    multiprocessing tasking to make use of multiple cores
    :param lst:
    :param process_func:
    :param num_procs:
    :param proc_desc:
    :param args:
    :param multi:
    :param file_filter_func:
    :param callback_func:
    :param thread_wise_objs:
    :param thread_init_func:
    :param thread_end_func:
    :return:
    """
    num_tasks = len(lst)
    pdq_queue = multiprocessing.JoinableQueue(num_tasks)
    for item in lst:
        pdq_queue.put_nowait(item)
    thread_num = min(num_tasks, num_procs)
    arr = [process_func] if args is None else [process_func] + args
    arr.insert(0, pdq_queue)
    thread_objs = []
    for i in range(thread_num):
        tarr = arr[:]
        thread_obj = None
        if thread_wise_objs is not None and isinstance(thread_wise_objs, list):
            thread_obj = thread_wise_objs[i]
        if thread_obj is None and thread_init_func is not None:
            thread_obj = thread_init_func()
            thread_objs.append(thread_obj)
        tarr.insert(0, thread_obj)
        p = multiprocessing.Process(target=multi_process_do, args=tuple(tarr))
        p.start()

    for i in range(thread_num):
        pdq_queue.put('EMPTY-NOW')
    pdq_queue.join()
    if thread_end_func is not None:
        for to in thread_objs:
            if to is not None:
                thread_end_func(to, *tuple(thread_end_args))
    if callback_func is not None:
        callback_func(*tuple(args))


def multi_process_large_file_tasking(large_file, process_func, num_procs=multiprocessing.cpu_count()*2,
                                     proc_desc='processed', args=None, multi=None,
                                     file_filter_func=None, callback_func=None,
                                     thread_wise_objs=None,
                                     thread_init_func=None, thread_end_func=None,
                                     file_encoding='utf-8', thread_end_args=[]):
    num_tasks = 1000
    pdf_queque = multiprocessing.JoinableQueue(num_tasks)
    thread_num = min(num_tasks, num_procs)
    arr = [process_func] if args is None else [process_func] + args
    arr.insert(0, pdf_queque)
    thread_objs = []
    for i in range(thread_num):
        tarr = arr[:]
        thread_obj = None
        if thread_wise_objs is not None and isinstance(thread_wise_objs, list):
            thread_obj = thread_wise_objs[i]
        if thread_obj is None and thread_init_func is not None:
            thread_obj = thread_init_func()
            thread_objs.append(thread_obj)
        tarr.insert(0, thread_obj)
        p = multiprocessing.Process(target=multi_process_do, args=tuple(tarr))
        p.start()

    num_lines = 0
    with codecs.open(large_file, encoding=file_encoding) as lf:
        for line in lf:
            num_lines += 1
            pdf_queque.put(line)
    for i in range(thread_num):
        pdf_queque.put('EMPTY-NOW')
    pdf_queque.join()
    if thread_end_func is not None:
        for to in thread_objs:
            if to is not None:
                thread_end_func(to, *tuple(thread_end_args))
    if callback_func is not None:
        callback_func(*tuple(args))


def setup_basic_logging(log_level='INFO', log_format='%(name)s %(asctime)s %(levelname)s %(message)s', file=None):
    logging.basicConfig(level=log_level, format=log_format)
    if file is not None:
        logging.basicConfig(filename=file, filemode='w', level=log_level, format=log_format)


if __name__ == "__main__":
    pass
