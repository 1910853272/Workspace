<<<<<<< HEAD
package com.itheima.d3_annotation;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target(ElementType.METHOD) // 注解只能注解方法。
@Retention(RetentionPolicy.RUNTIME) // 让当前注解可以一直存活着。
public @interface MyTest {
}
=======
package com.itheima.d3_annotation;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target(ElementType.METHOD) // 注解只能注解方法。
@Retention(RetentionPolicy.RUNTIME) // 让当前注解可以一直存活着。
public @interface MyTest {
}
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
