package com.itheima.entry;

import com.itheima.thread.CompileRunnable;
import com.itheima.thread.ExecRunnable;

public class Entry {

    public static void main(String[] args) {

        // 定义要热加载的java源文件所对应的包
        String hotLoaderSourceFilePackage = "com.itheima.domain" ;

        // 开启一个线程每隔1秒检测Worker.java源文件个变化情况，如果发生了改变就从新进行编译
        new Thread(new CompileRunnable(hotLoaderSourceFilePackage)).start();

        // 开启一个线程每隔1秒执行一次热更新目录下的所有的类中的所有方法(只考虑无参数的方法)
        new Thread(new ExecRunnable(hotLoaderSourceFilePackage)).start();


    }

}
