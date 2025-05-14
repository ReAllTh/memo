# 抽烟者问题

问题描述：

> 有一个供应者和三个抽烟者。每个抽烟者不停地卷烟并抽掉它，但要卷起并抽掉一支烟，抽烟者需要有三种材料：烟草、纸和胶水。三个抽烟者中第一个拥有烟草，第二个拥有纸，第三个拥有胶水。供应者进程无限地供应三种材料，供应者每次将两种材料放到桌子上，拥有剩下那种材料的的抽烟者卷起一根烟并抽掉它，并给供应者一个信号告诉已完成，此时供应者就会将另外两种材料放到桌上，如此重复（让三个抽烟者轮流地抽烟）。

固定解法。如下：

```c
semaphore offer1 = 0;
semaphore offer2 = 0;
semaphore offer3 = 0;
semaphore finish = 0;

int num = 0;

Suplyer() {
	while (1) {
		++num;
		num = num % 3;
		
		if (num == 0) {
			V(offer1);
		} else if (num == 1) {
			V(offer2);
		} else {
			V(offer3);
		}
		
		P(finish);
	}
}

Smoker1() {
	while (1) {
		P(offer1);
		smoke();
		V(finish);
	}
}

Smoker2() {
	while (1) {
		P(offer2);
		smoke();
		V(finish);
	}
}

Smoker3() {
	while (1) {
		P(offer3);
		smoke();
		V(finish);
	}
}
```

