<<<<<<< HEAD
package com.itheima.d3_annotation;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target({ElementType.TYPE,ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
public @interface MyTest4 {
    String value();
    double aaa() default 100;
    String[] bbb();
}
=======
package com.itheima.d3_annotation;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target({ElementType.TYPE,ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
public @interface MyTest4 {
    String value();
    double aaa() default 100;
    String[] bbb();
}
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
