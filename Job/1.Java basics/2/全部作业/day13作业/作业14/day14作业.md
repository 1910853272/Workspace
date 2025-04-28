##  day14作业

#### 一、今日单词

1. 反射中所涉及的类名以及方法名称：

   ```java
   答：
   ```

2. 注解以及元注解中所涉及的单词：

   ```java
   答：
   ```

3. 动态代理中所涉及的类名及单词：

   ```java
   答：
   ```

------

#### 二、问答题

1. 什么是单元测试？使用步骤？
   ```java
   答：
   ```
2. 什么是反射？有什么用？
   ```java
   答：
   ```
3. 什么是注解？注解属性的类型？
   ```java
   答：
   ```
4. 什么是动态代理？
   ```java
   答：
   ```
------
#### 三、 代码题：

##### 第一题：分析以下需求，并用代码实现

**训练目标**：

​	掌握java中反射的基本使用

**需求描述**：

1. ArrayList<Integer> list = new ArrayList<Integer>(); 

   这个泛型为Integer的ArrayList中存放一个String类型的对象


------

##### 第二题：分析以下需求，并用代码实现

**训练目标**：

​	掌握java中反射的基本使用

**需求描述**：

​	定义一个标准的JavaBean，名叫Person，包含属性name、age。

​	使用反射的方式创建一个实例、调用构造函数初始化name、age，使用反射方式调用setName方法对名称进行设置，不使用setAge方法直接使用反射方式对age赋值。

------

##### 第三题：分析以下需求，并用代码实现

**训练目标**：

​	掌握java中反射的基本使用

**需求描述**：

用反射的方式运行run方法。

已知一个类，定义如下： 
	package com.itheima; 
	public class DemoClass { 
		public void run() { 
			System.out.println("welcome to heima!"); 
		} 
	} 
	(1)写一个Properties格式的配置文件，配置类的完整名称。
	(2) 写一个程序，读取这个Properties配置文件，获得类的完整名称并加载这个类，
------

##### 第四题：分析以下需求，并用代码实现

**训练目标**：

​	掌握java中反射的基本使用

**需求描述**：

写一个方法，此方法可以获取obj对象中名为propertyName的属性的值   
	public Object getProperty(Object obj, String propertyName){ 
		// 完成代码
	}
------

##### 第五题：分析以下需求，并用代码实现

**训练目标**：

​	掌握java中反射和注解的基本使用

**需求描述**：

使用注解替代xml功能：

1. 自定义注解Test，并定义三个属性：className（全类名）、methodName（方法名）、paramValue（参数值）
2. 定义测试类，在测试上使用该注解，并给注解三个属性赋值
3. 通过返回获取该注解的属性值，并能成功创建className指定类的对象
4. 调用methodName的方法，并传入paramValue的参数值

   ​

   ​














