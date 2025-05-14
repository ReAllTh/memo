# 生产者-消费者问题

## 单生产者、单消费者、单缓冲区（经典生产者消费者问题）

题目描述：

> 一组生产者进程和一组消费者进程共享一个初始为空、大小为  $$ n $$  的缓冲区，只有缓冲区没满时，生产者才能把产品放入缓冲区，否则必须等待；只有缓冲区不为空时，消费者才能从中取出产品，否则必须等待。由于缓冲区是临界资源，它只允许一个生产者放入产品，或一个消费者从中取出产品。

经典问题，固定解法。如下：

```c
semaphore empty = n; // 缓冲区空位数
semaphore full = 0; // 缓冲区产品数
semaphore mutex = 1; // 缓冲区互斥锁

Producer() {
	while (1) {
		P(empty);
		P(mutex);
		
		put();
		
		V(mutex);
		V(full);
	}
}

Consumer() {
	while (1) {
		P(full);
		P(mutex);
		
		consume();
		
		V(mutex);
		V(empty);
	}
}
```

程序中，`P(mutex)` 和 `V(mutex)` 必须成对出现，加载二者之间的代码段是临界区；施加于信号量 `empty` 和 `full` 的 PV 操作也必须成对出现，但分别位于不同的程序中。

在使用信号量和 PV 操作进行进程同步时，P 操作的次序是非常重要的，而 V 操作的次序无关紧要。例如：如果把 `Producer` 进程中的 P 操作交换次序，在缓冲区先行占满时会导致死锁。

一般来说，互斥信号量的 P 操作总是在后面执行。

另外，如果题目没有要求 “取的时候不能拿，拿的时候也不能取” 的话，可以把 `mutex` 拆成 `mutex_producer` 和 `mutex_consumer`，分别用于生产者进程和消费者进程各自之间的互斥，这样可以有更高的并发性。

## 双生产者、双消费者、单缓冲区（水果问题）

题目描述：

> 桌子上有一个盘子，每次只能向其中放入一个水果。爸爸专向盘子中放苹果，妈妈专向盘子中放橘子，儿子专等吃盘子中的橘子，女儿专等吃盘子中的苹果。只有盘子为空时，爸爸或妈妈才可向盘子中放一个水果；仅当盘子中有自己需要的水果时，儿子或女儿可以从盘子中取出。

```c
semaphore apple = 0; // 苹果数量
semaphore orange = 0; // 橘子数量
semaphore plate = 1; // 盘子互斥锁

dad() {
	while (1) {
		P(plate);
		put_apple();
		V(apple);
	}
}

mom() {
	while (1) {
		P(plate);
		put_orange();
		V(orange);
	}
}

son() {
	while (1) {
		P(apple);
		eat_apple();
		V(plate);
	}
}

daughter() {
	while (1) {
		P(orange);
		eat_orange();
		V(plate);
	}
}
```

## 双生产者、单消费者、双缓冲区（装配车间问题）

题目描述：

> 工厂有两个生产车间和一个装配车间，两个生产车间分别生产零件 A 和零件 B，装配车间的任务是把这两种零件组装成产品。两个生产车间每生产一个零件都要分别把它们送到装配车间的货架 F~1~ 和 F~2~ 上，F~1~ 存放零件 A，F~2~ 存放零件 B，且 F~1~ 和 F~2~ 均只能容纳  $$ 10 $$  个零件。装配车间每当能从货架上取到一个零件 A 和一个零件 B 后就将它们组装成一件产品。使用 P、V 操作进行管理，使各车间相互合作、协调工作。

解法：

```c
semaphore empty_F1 = 10; // 货架 F1 空位数
semaphore empty_F2 = 10; // 货架 F2 空位数
semaphore full_F1_A = 0; // 零件 A 的数量
semaphore full_F2_B = 0; // 零件 B 的数量
semaphore mutex_F1 = 1; // 货架 F1 互斥锁
semaphore mutex_F2 = 1; // 货架 F2 互斥锁

Producer_A() {
	while (1) {
		produce_A();
		
		P(empty_F1);
		P(mutex_F1);
		put_A();
		V(mutex_F1);
		V(full_F1_A);
	}
}

Producer_B() {
	while (1) {
		produce_B();
		
		P(empty_F2);
		P(mutex_F2);
		put_B();
		V(mutex_F2);
		V(full_F2_B);		
	}
}

Loader() {
	while (1) {
		P(full_F1_A);
		P(mutex_F1);
		get_A();
		V(mutex_F1);
		V(empty_F1);
		
		P(full_F2_B);
		P(mutex_F2);
		get_B();
		V(mutex_F2);
		V(empty_F2);
		
		load();	
	}
}
```

## 双生产者、单消费者、单缓冲区（自行车产线问题）

题目描述：

> 自行车生产线上有一个箱子，其中有  $$ N $$  个位置（ $$ N ≥ 3 $$ ），每个位置可存放一个车架或一个车轮，设有  $$ 3 $$  名工人，其活动分别为：
>
> ```pseudocode
> 工人 1() {
>     while (1) {
>         加工一个车架
>         车架放入箱中
>     }
> }
> 
> 工人 2() {
>     while (1) {
>         加工一个车轮
>         车轮放入箱中
>     }
> }
> 
> 工人 3() {
>     while (1) {
>         箱中取出一个车架
>         箱中取出两个车轮
>         组装为一台车
>     }
> }
> ```
>
> 试分别用信号量与 PV 操作实现三名工人的合作，要求解中不含死锁。

解法：

```c
semaphore empty_box = N;
semaphore empty_frame = N - 2;
semaphore empty_wheel = N - 1;
semaphore full_frame = 0;
semaphore full_wheel = 0;

worker_1(){
    while (1) {
        加工一个车架
        
        P(empty_box);
        P(empty_frame);
        车架放入箱中
        V(full_frame);
    }
}

worker_2() {
    while (1) {
        加工一个车轮
        
        P(empty_box);
        P(empty_wheel);
        车轮放入箱中
        V(full_wheel);
    }
}

worker_3() {
    while (1) {
        P(full_frame);
        箱中取出一个车架
        V(empty_frame);
        V(empty_box);

        P(full_wheel);
        P(full_wheel);
        箱中取出二个车轮
        V(empty_wheel);
        V(empty_wheel);
        V(empty_box);
        V(empty_box);
        
        组装为一台车
    }
}
```

