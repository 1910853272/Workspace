<<<<<<< HEAD
package com.itheima.reflect.context;

import java.io.InputStream;
import java.util.Collection;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

public class ClasspathApplicationContext implements ApplicationContext {

    // 定义一个容器存储系统中可能使用到的对象
    private ConcurrentHashMap<String, Object> beans = new ConcurrentHashMap<>();

    // 定义构造方法
    public ClasspathApplicationContext(String fileName) {

        // 补全代码
        try {

            // 读取配置文件中的内容，将其存储到Properties集合中
            Properties properties = new Properties() ;
            InputStream inputStream = ClasspathApplicationContext.class.getClassLoader().getResourceAsStream(fileName);
            properties.load(inputStream);

            // 遍历集合，通过反射创建对象，存储到beans集合中
            Set<String> names = properties.stringPropertyNames();
            for(String name : names) {
                String className = properties.getProperty(name);
                Class clazz = Class.forName(className);
                beans.put(name , clazz.newInstance()) ;
            }

        }catch (Exception e) {
            e.printStackTrace();
        }

    }

    @Override
    public <T> T getBean(Class<T> clazz) {

        // 补全代码
        // 遍历beans集合
        Collection<Object> values = beans.values();
        for(Object obj : values) {
            if(clazz == obj.getClass()) {       // 判断Class类型是否相当，如果是直接返回该对象即可
                return (T)obj ;
            }
        }

        return null;
    }

}
=======
package com.itheima.reflect.context;

import java.io.InputStream;
import java.util.Collection;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

public class ClasspathApplicationContext implements ApplicationContext {

    // 定义一个容器存储系统中可能使用到的对象
    private ConcurrentHashMap<String, Object> beans = new ConcurrentHashMap<>();

    // 定义构造方法
    public ClasspathApplicationContext(String fileName) {

        // 补全代码
        try {

            // 读取配置文件中的内容，将其存储到Properties集合中
            Properties properties = new Properties() ;
            InputStream inputStream = ClasspathApplicationContext.class.getClassLoader().getResourceAsStream(fileName);
            properties.load(inputStream);

            // 遍历集合，通过反射创建对象，存储到beans集合中
            Set<String> names = properties.stringPropertyNames();
            for(String name : names) {
                String className = properties.getProperty(name);
                Class clazz = Class.forName(className);
                beans.put(name , clazz.newInstance()) ;
            }

        }catch (Exception e) {
            e.printStackTrace();
        }

    }

    @Override
    public <T> T getBean(Class<T> clazz) {

        // 补全代码
        // 遍历beans集合
        Collection<Object> values = beans.values();
        for(Object obj : values) {
            if(clazz == obj.getClass()) {       // 判断Class类型是否相当，如果是直接返回该对象即可
                return (T)obj ;
            }
        }

        return null;
    }

}
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
