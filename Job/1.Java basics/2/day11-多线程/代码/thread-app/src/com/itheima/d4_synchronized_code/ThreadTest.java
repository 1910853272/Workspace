<<<<<<< HEAD
package com.itheima.d4_synchronized_code;

/**
 * 目标：模拟线程安全问题。
 */
public class ThreadTest {
    public static void main(String[] args) {
        Account acc = new Account("ICBC-110", 100000);
        new DrawThread(acc, "小明").start(); // 小明
        new DrawThread(acc, "小红").start(); // 小红

        Account acc1 = new Account("ICBC-112", 100000);
        new DrawThread(acc1, "小黑").start(); // 小黑
        new DrawThread(acc1, "小白").start(); // 小白
    }
}
=======
package com.itheima.d4_synchronized_code;

/**
 * 目标：模拟线程安全问题。
 */
public class ThreadTest {
    public static void main(String[] args) {
        Account acc = new Account("ICBC-110", 100000);
        new DrawThread(acc, "小明").start(); // 小明
        new DrawThread(acc, "小红").start(); // 小红

        Account acc1 = new Account("ICBC-112", 100000);
        new DrawThread(acc1, "小黑").start(); // 小黑
        new DrawThread(acc1, "小白").start(); // 小白
    }
}
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
