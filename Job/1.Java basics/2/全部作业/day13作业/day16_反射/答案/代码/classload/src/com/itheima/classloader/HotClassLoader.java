<<<<<<< HEAD
package com.itheima.classloader;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

// 自定义热更新的类加载器
public class HotClassLoader extends ClassLoader {

    // 补全代码
    private String classFilePath ;
    public HotClassLoader(String classFilePath) {
        this.classFilePath = classFilePath ;
    }


    @Override
    public Class<?> loadClass(String name) throws ClassNotFoundException {

        // 补全代码
        try {

            // 判断要加载的类是否是自定义的类
            if(name.startsWith("com.itheima.domain")) {

                // 读取字节码文件数据
                File classFile = new File(classFilePath) ;
                InputStream inputStream = new FileInputStream(classFile) ;
                byte[] classByteArr = new byte[(int)classFile.length()] ;
                inputStream.read(classByteArr) ;

                // 调用父类的defineClass方法将字节数组加载到JVM中
                return defineClass(name , classByteArr , 0 , classByteArr.length) ;

            }else {
                return super.loadClass(name) ;      // 调用父类加载器完成加载
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

}
=======
package com.itheima.classloader;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

// 自定义热更新的类加载器
public class HotClassLoader extends ClassLoader {

    // 补全代码
    private String classFilePath ;
    public HotClassLoader(String classFilePath) {
        this.classFilePath = classFilePath ;
    }


    @Override
    public Class<?> loadClass(String name) throws ClassNotFoundException {

        // 补全代码
        try {

            // 判断要加载的类是否是自定义的类
            if(name.startsWith("com.itheima.domain")) {

                // 读取字节码文件数据
                File classFile = new File(classFilePath) ;
                InputStream inputStream = new FileInputStream(classFile) ;
                byte[] classByteArr = new byte[(int)classFile.length()] ;
                inputStream.read(classByteArr) ;

                // 调用父类的defineClass方法将字节数组加载到JVM中
                return defineClass(name , classByteArr , 0 , classByteArr.length) ;

            }else {
                return super.loadClass(name) ;      // 调用父类加载器完成加载
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

}
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
