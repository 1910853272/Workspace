<<<<<<< HEAD
package com.itheima.d5_block;

public class Student {
    static int number = 80;
    static String schoolName = "黑马";
    // 静态代码块
    static {
        System.out.println("静态代码块执行了~~");
        // schoolName = "黑马";
    }
    int age;
    // 实例代码块
    {
        System.out.println("实例代码块执行了~~");
        // age = 18;
        System.out.println("有人创建了对象：" + this);
    }

    public Student(){
        System.out.println("无参数构造器执行了~~");
    }

    public Student(String name){
        System.out.println("有参数构造器执行了~~");
    }
}



=======
package com.itheima.d5_block;

public class Student {
    static int number = 80;
    static String schoolName = "黑马";
    // 静态代码块
    static {
        System.out.println("静态代码块执行了~~");
        // schoolName = "黑马";
    }
    int age;
    // 实例代码块
    {
        System.out.println("实例代码块执行了~~");
        // age = 18;
        System.out.println("有人创建了对象：" + this);
    }

    public Student(){
        System.out.println("无参数构造器执行了~~");
    }

    public Student(String name){
        System.out.println("有参数构造器执行了~~");
    }
}



>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
