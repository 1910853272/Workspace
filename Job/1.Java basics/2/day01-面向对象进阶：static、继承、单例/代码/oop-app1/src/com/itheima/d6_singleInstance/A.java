<<<<<<< HEAD
package com.itheima.d6_singleInstance;

public class A {
    // 2、定义一个类变量记住类的一个对象
    private static A a = new A();

    // 1、必须私有类的构造器
    private A(){

    }

    // 3、定义一个类方法返回类的对象
    public static A getObject(){
        return a;
    }
}
=======
package com.itheima.d6_singleInstance;

public class A {
    // 2、定义一个类变量记住类的一个对象
    private static A a = new A();

    // 1、必须私有类的构造器
    private A(){

    }

    // 3、定义一个类方法返回类的对象
    public static A getObject(){
        return a;
    }
}
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
