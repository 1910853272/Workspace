<<<<<<< HEAD
package com.itheima.d8_interface2;


import com.itheima.d5_abstract2.Animal;
import com.itheima.d5_abstract2.Cat;

public class Test {
    public static void main(String[] args) {
        // 目标：搞清楚使用接口的好处。
        Driver s = new A();
        s.drive();

        Driver d = new A();
        d.drive();

        Animal a = new Cat();
        a.cry();
    }
}

class B implements Driver{
    @Override
    public void drive() {

    }
}

class A extends Student implements Driver, Singer{
    @Override
    public void drive() {

    }

    @Override
    public void sing() {

    }
}

class Student{

}

interface Driver{
    void drive();
}

interface Singer{
    void sing();
=======
package com.itheima.d8_interface2;


import com.itheima.d5_abstract2.Animal;
import com.itheima.d5_abstract2.Cat;

public class Test {
    public static void main(String[] args) {
        // 目标：搞清楚使用接口的好处。
        Driver s = new A();
        s.drive();

        Driver d = new A();
        d.drive();

        Animal a = new Cat();
        a.cry();
    }
}

class B implements Driver{
    @Override
    public void drive() {

    }
}

class A extends Student implements Driver, Singer{
    @Override
    public void drive() {

    }

    @Override
    public void sing() {

    }
}

class Student{

}

interface Driver{
    void drive();
}

interface Singer{
    void sing();
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
}