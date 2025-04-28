<<<<<<< HEAD
package com.itheima.d2_reflect;

/**
 * 目标：获取Class对象。
 */
public class Test1Class {
    public static void main(String[] args) throws Exception {
        Class c1 = Student.class;
        System.out.println(c1.getName()); // 全类名
        System.out.println(c1.getSimpleName()); // 简名：Student

        Class c2 = Class.forName("com.itheima.d2_reflect.Student");
        System.out.println(c1 == c2);

        Student s = new Student();
        Class c3 = s.getClass();
        System.out.println(c3 == c2);
    }
}
=======
package com.itheima.d2_reflect;

/**
 * 目标：获取Class对象。
 */
public class Test1Class {
    public static void main(String[] args) throws Exception {
        Class c1 = Student.class;
        System.out.println(c1.getName()); // 全类名
        System.out.println(c1.getSimpleName()); // 简名：Student

        Class c2 = Class.forName("com.itheima.d2_reflect.Student");
        System.out.println(c1 == c2);

        Student s = new Student();
        Class c3 = s.getClass();
        System.out.println(c3 == c2);
    }
}
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
