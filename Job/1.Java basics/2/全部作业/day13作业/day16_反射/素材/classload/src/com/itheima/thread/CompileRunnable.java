package com.itheima.thread;

import javax.tools.JavaCompiler;
import javax.tools.ToolProvider;
import java.io.File;

public class CompileRunnable implements Runnable {

    // 定义要热加载的源文件的包,并通过构造方法进行初始化
    private String hotLoaderSourceFilePackage ;
    public CompileRunnable(String hotLoaderSourceFilePackage) {
        this.hotLoaderSourceFilePackage = hotLoaderSourceFilePackage ;
    }

    @Override
    public void run() {

        // 定义循环，每隔一秒执行一次
        while(true) {

            // 获取热加载下的源文件的目录中所有的文件
            String sourceFileDirectory = "src/" + hotLoaderSourceFilePackage.replace("." , "/") + "/" ;
            File directory = new File(sourceFileDirectory);
            File[] files = directory.listFiles();
            for(File file : files) {
                if(file.getName().endsWith(".java")) {
                    JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();       // 动态编译
                    int state = compiler.run(null, null, null, sourceFileDirectory + file.getName());
                    if(state != 0) {
                        System.out.println("编译" + file.getName() + "失败了.........");
                    }else {
                        System.out.println("编译" + file.getName() + "成功了.........");
                    }
                }
            }

            try {
                Thread.sleep(1000);         // 线程休眠1秒
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

        }

    }

}
