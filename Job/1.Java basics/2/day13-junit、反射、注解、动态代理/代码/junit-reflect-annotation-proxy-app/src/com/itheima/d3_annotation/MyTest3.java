<<<<<<< HEAD
package com.itheima.d3_annotation;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target({ElementType.TYPE, ElementType.METHOD}) // 当前被修饰的注解只能用在类上，方法上。
@Retention(RetentionPolicy.RUNTIME) // 控制下面的注解一直保留到运行时
public @interface MyTest3 {
}
=======
package com.itheima.d3_annotation;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target({ElementType.TYPE, ElementType.METHOD}) // 当前被修饰的注解只能用在类上，方法上。
@Retention(RetentionPolicy.RUNTIME) // 控制下面的注解一直保留到运行时
public @interface MyTest3 {
}
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
