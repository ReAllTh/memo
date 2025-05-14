# 读者-写者问题

问题描述：

> 有读者和写者两组并发进程，共享一个文件，当多个读进程同时访问共享数据时不会产生副作用，但若某个写进程和其他进程（读进程或写进程）同时访问共享数据时，则可能导致数据不一致的错误。因此要求：
>
> 1. 允许多个读者同时对文件执行读操作；
> 2. 只允许一个写者对文件执行写操作；
> 3. 任何写者在完成写操作前不允许其他读者或写者工作；
> 4. 写者在执行写操作前，应让已有的写者和读者全部退出。

有三种解法：

1. 读进程优先解法，如下：

    ```c
    semaphore rw = 1;
    semaphore mutex = 1;
    
    int count = 0;
    
    Reader() {
        while (1) {
            P(mutex);
            if (count == 0) {
                P(rw);
            }
            ++count;
            V(mutex);
    
            read();
    
            P(mutex);
            --count;
            if (count == 0) {
                V(rw);
            }
            V(mutex);
        }
    }
    
    Writer() {
        while (1) {
            P(rw);
    
            write();
    
            V(rw);
        }
    }
    ```
	
	这种解法中读进程是优先的，当存在读者时，写者将被延迟，且只要有一个读者活跃，随后而来的读者都将被允许访问文件，从而导致写者长时间等待，并有可能出现写者饥饿现象。

2. 写进程优先解法不会发生饥饿，简单概括就是在获取 `rw` 这个不公平的锁之前获取一个公平的锁 `w`，如下：

    ```c
    semaphore w = 1;
    semaphore rw = 1;
    semaphore mutex = 1;
	
    int count = 0;
	
    Reader() {
        while (1) {
            P(w);
            P(mutex);
            if (count == 0) {
                P(rw);
            }
            ++count;
            V(mutex);
            V(w);
	
            read();
	
            P(mutex);
            --count;
            if (count == 0) {
                V(rw);
            }
            V(mutex);
        }
    }
	
    Writer() {
        while (1) {
            P(w);
            P(rw);
	
            write();
	
            V(rw);
            V(w);
        }
    }
    ```

	这里的写进程优先是相对而言的，有些书上把这个算法称为 “读写公平法”，即读写进程具有一样的优先级。

3. “真正的写者” 优先会导致饥饿且比较复杂，考试一般不考，如下：

    ```c
    semaphore read = 1; // 代表临界资源的读取权
    semaphore write = 1; // 代表临界资源的写入权
    semaphore mutex_count_reader = 1; // 对变量 count_reader 的互斥锁
    semaphore mutex_count_writer = 1; // 对变量 count_writer 的互斥锁
    
    int count_reader = 0; // 用于记录正在读取的读者数量
    int count_writer = 0; // 用于记录即将要写入的写者数量
    
    Reader() {
        while (1) {
            P(read); // 每个读进程都需要获取临界资源的读取权
            P(mutex_count_reader);
            if (count_reader == 0) { // 为了阻塞后续的写进程，第一个读进程要占有临界资源的写入权
                P(write);
            }
            ++count_reader;
            V(mutex_count_reader);
    
            V(read); // 为了保证读进程同时访问，在这里把读取权让给后续的读进程
            读数据
    
            P(mutex_count_reader);
            --count_reader; 
            if (count_reader == 0) { // 为了唤醒后续写进程，最后一个读进程要让出临界资源的写入权
                V(write);
            }
            V(mutex_count_reader);
        }
    }
    
    Writer() {
        while (1) {
            P(mutex_count_writer);
            if (count_writer == 0) { // 为了阻塞后续的读进程，第一个写进程要占有临界资源的读取权
                P(read);
            }
            ++count_writer;
            V(mutex_count_writer);
    
            P(write); // 获取临界资源的访问权
            写数据
            V(write); // 写入数据完成，让出临界资源的写入权
    
            P(mutex_count_writer);
            --count_writer;
            if (count_writer == 0) { // 为了唤醒后续读进程，最后一个写进程要让出临界资源的写入权
                V(read);
            }
            V(mutex_count_writer)
        }
    }
    ```

