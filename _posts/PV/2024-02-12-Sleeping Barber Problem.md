# 睡眠理发师问题

## 经典睡眠理发师问题

问题描述：

> 理发店有一位理发师、一把理发椅和  $$ n $$  把供等候理发的顾客休憩的椅子；如果没有顾客，理发师便在理发椅上睡觉，当有顾客到来时，他唤醒理发师；如果理发师正在理发时又有新顾客来到，那么，如果还有空椅子，顾客就坐下来等待，否则就离开理发店。

经典问题，固定解法。如下：

```c
semaphore barber = 0;
semaphore customer = 0;
semaphore mutex = 1;

int waiting = 0;

Barber() {
	while (1) {
		P(customer);
		P(mutex);
		--waiting;
		V(barber);
		V(mutex);
		
		cuthair();
	}
}

Customer_i() {
	P(mutex);
	if (waiting < n) {
		++waiting;
		V(customer);
		V(mutex);
		P(barber);
		
		get_haircut();
	} else {
		V(mutex);
	}
}
```

