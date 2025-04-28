<<<<<<< HEAD
package com.itheima.thisdemo;

public class Test {
    public static void main(String[] args) {
        // 目标：认识this，掌握this的应用场景。
        Student s1 = new Student();
        System.out.println(s1);
        s1.test();

        Student s2 = new Student();
        System.out.println(s2);
        s2.test();

        System.out.println("----------------------------");

        Student s3 = new Student();
        s3.name = "播仔";
        s3.score = 2;
        s3.printPass(256);
    }
}
=======
package com.itheima.thisdemo;

public class Test {
    public static void main(String[] args) {
        // 目标：认识this，掌握this的应用场景。
        Student s1 = new Student();
        System.out.println(s1);
        s1.test();

        Student s2 = new Student();
        System.out.println(s2);
        s2.test();

        System.out.println("----------------------------");

        Student s3 = new Student();
        s3.name = "播仔";
        s3.score = 2;
        s3.printPass(256);
    }
}
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
