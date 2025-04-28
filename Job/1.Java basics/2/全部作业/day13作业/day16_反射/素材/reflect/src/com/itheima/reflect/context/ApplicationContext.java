package com.itheima.reflect.context;

public interface ApplicationContext {

    // 根据Class类型从容器中获取指定的对象
    public abstract <T> T getBean(Class<T> clazz) ;

}
