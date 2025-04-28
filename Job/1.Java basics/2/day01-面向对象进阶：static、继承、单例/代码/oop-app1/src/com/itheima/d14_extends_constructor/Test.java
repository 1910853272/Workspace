<<<<<<< HEAD
package com.itheima.d14_extends_constructor;

class F{
//    public F(){
//        System.out.println("===父类F的 无参数构造器 执行了===");
//    }

    public F(String name, int age){

    }
}

class Z extends F{
    public Z(){
        // super(); // 默认存在的
        super("播妞", 17);
        System.out.println("===子类Z的 无参数构造器 执行了===");
    }

    public Z(String name){
        // super(); // 默认存在的
        super("播妞", 17);
        System.out.println("===子类Z的 有参数构造器 执行了===");
    }
}

public class Test {
    public static void main(String[] args) {
        // 目标：先认识子类构造器的特点，再掌握这个特点的常见应用场景。
        Z z = new Z();
        Z z2 = new Z("播妞");
    }
}
=======
package com.itheima.d14_extends_constructor;

class F{
//    public F(){
//        System.out.println("===父类F的 无参数构造器 执行了===");
//    }

    public F(String name, int age){

    }
}

class Z extends F{
    public Z(){
        // super(); // 默认存在的
        super("播妞", 17);
        System.out.println("===子类Z的 无参数构造器 执行了===");
    }

    public Z(String name){
        // super(); // 默认存在的
        super("播妞", 17);
        System.out.println("===子类Z的 有参数构造器 执行了===");
    }
}

public class Test {
    public static void main(String[] args) {
        // 目标：先认识子类构造器的特点，再掌握这个特点的常见应用场景。
        Z z = new Z();
        Z z2 = new Z("播妞");
    }
}
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
