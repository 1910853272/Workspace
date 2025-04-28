<<<<<<< HEAD
package com.itheima.demo3.Timer;

import java.util.Timer;

public class Start {

    public static void main(String[] args) {
        // 创建一个定时器对象
        Timer timer = new Timer() ;
        timer.schedule(new TimeTask(), 0 , 1000);         // 每隔1秒执行一次new TimeTask()里的run方法
    }
}
=======
package com.itheima.demo3.Timer;

import java.util.Timer;

public class Start {

    public static void main(String[] args) {
        // 创建一个定时器对象
        Timer timer = new Timer() ;
        timer.schedule(new TimeTask(), 0 , 1000);         // 每隔1秒执行一次new TimeTask()里的run方法
    }
}
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
