<<<<<<< HEAD
package com.itheima.reflect.entry;

import com.itheima.reflect.context.ApplicationContext;
import com.itheima.reflect.context.ClasspathApplicationContext;
import com.itheima.reflect.service.OrderService;
import com.itheima.reflect.service.UserService;

public class Entry {

    public static void main(String[] args) {

        // 创建ClasspathApplicationContext对象
        ApplicationContext applicationContext = new ClasspathApplicationContext("applicationContext.properties") ;

        // 从applicationContext对象获取userService对象，并进行使用
        UserService userService = applicationContext.getBean(UserService.class);
        userService.find();

        // 从applicationContext对象获取OrderService对象，并进行使用
        OrderService orderService1 = applicationContext.getBean(OrderService.class);
        orderService1.find();

        // 从applicationContext对象获取OrderService对象，比较与上面所获取的对象是否相等
        OrderService orderService2 = applicationContext.getBean(OrderService.class);
        System.out.println(orderService1 == orderService2);
    }

}
=======
package com.itheima.reflect.entry;

import com.itheima.reflect.context.ApplicationContext;
import com.itheima.reflect.context.ClasspathApplicationContext;
import com.itheima.reflect.service.OrderService;
import com.itheima.reflect.service.UserService;

public class Entry {

    public static void main(String[] args) {

        // 创建ClasspathApplicationContext对象
        ApplicationContext applicationContext = new ClasspathApplicationContext("applicationContext.properties") ;

        // 从applicationContext对象获取userService对象，并进行使用
        UserService userService = applicationContext.getBean(UserService.class);
        userService.find();

        // 从applicationContext对象获取OrderService对象，并进行使用
        OrderService orderService1 = applicationContext.getBean(OrderService.class);
        orderService1.find();

        // 从applicationContext对象获取OrderService对象，比较与上面所获取的对象是否相等
        OrderService orderService2 = applicationContext.getBean(OrderService.class);
        System.out.println(orderService1 == orderService2);
    }

}
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
