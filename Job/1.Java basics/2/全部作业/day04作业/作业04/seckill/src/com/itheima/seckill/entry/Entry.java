<<<<<<< HEAD
package com.itheima.seckill.entry;

import com.itheima.seckill.task.TimeTask;

import java.util.Timer;

public class Entry {

    public static void main(String[] args) {

        // 创建一个定时器对象
        Timer timer = new Timer() ;
        timer.schedule(new TimeTask(), 0 , 1000);         // 每隔1秒执行一次

    }

}
=======
package com.itheima.seckill.entry;

import com.itheima.seckill.task.TimeTask;

import java.util.Timer;

public class Entry {

    public static void main(String[] args) {

        // 创建一个定时器对象
        Timer timer = new Timer() ;
        timer.schedule(new TimeTask(), 0 , 1000);         // 每隔1秒执行一次

    }

}
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
