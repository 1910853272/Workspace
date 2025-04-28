<<<<<<< HEAD
package com.itheima.d4_abstract;
// 抽象类
public abstract class A {
    private String name;
    public static String schoolName;

    public A(){
    }

    public A(String name) {
        this.name = name;
    }

    // 抽象方法：必须用abstract修饰，只有方法签名，一定不能有方法体
    public abstract void run();

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }
}

=======
package com.itheima.d4_abstract;
// 抽象类
public abstract class A {
    private String name;
    public static String schoolName;

    public A(){
    }

    public A(String name) {
        this.name = name;
    }

    // 抽象方法：必须用abstract修饰，只有方法签名，一定不能有方法体
    public abstract void run();

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }
}

>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
