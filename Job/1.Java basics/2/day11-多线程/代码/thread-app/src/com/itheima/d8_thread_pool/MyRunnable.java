<<<<<<< HEAD
package com.itheima.d8_thread_pool;

public class MyRunnable implements Runnable{
    @Override
    public void run() {
        // 任务是干啥的？
        System.out.println(Thread.currentThread().getName() + " ==> 输出666~~");
        try {
            Thread.sleep(Integer.MAX_VALUE);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
=======
package com.itheima.d8_thread_pool;

public class MyRunnable implements Runnable{
    @Override
    public void run() {
        // 任务是干啥的？
        System.out.println(Thread.currentThread().getName() + " ==> 输出666~~");
        try {
            Thread.sleep(Integer.MAX_VALUE);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
