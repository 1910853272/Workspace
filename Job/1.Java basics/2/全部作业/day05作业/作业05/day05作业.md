## day05作业

#### 一、选择题：

#####   1. 下列对于Lambda表达式的格式 ( 形式参数 ) -> { 代码块 ) 说法错误的是 ( B ) 

​    A. Lambda表达式必须作用在函数式接口上

​    B. 形式参数：如果有多个参数，参数之间用分号隔开；如果没有参数，留空即可

​    C. ->：由英文中画线和大于符号组成，固定写法。代表指向动作

​    D. 代码块：是我们具体要做的事情，也就是以前我们写的方法体内容


##### 2. 下列对于Lambda表达式的省略模式说法错误的是( D )

   A. 参数类型可以省略，但是有多个参数的情况下，不能只省略一个

   B. 如果参数有且仅有一个，那么小括号可以省略

   C. 如果代码块的语句只有一条，可以省略大括号和分号和return，不能只省略一个

   D. Lambda的指向箭头->可以省略

##### 3.【多选题】下列关于Lambda表达式说法正确的是（BCD）

   A.当一个接口中有多个抽象方法的时候，可以通过Lambda表达式来重写其中的方法

   B.可以通过::来传递Arrays.sort ()

   C.由于Lambda表达式是表示重写某个抽象方法，因此参数列表不可省略

   D.当Lambda表达式的函数体只有一句的时候，可以省略return不写

##### 4. 下列对于Arrays.sort()方法说法正确的是（C）

   A. sort(T[] arr,Comparator c)方法可以对任意类型数组排序

   B. sort(T[] arr,Comparator c)方法只能对数组进行升序排序

   C. sort(T[] arr,Comparator c)方法只能对引用类型数组排序

   D. sort()方法没有重载，参数是Object，就可以接受任意类型数组

##### 5.  【多选题】下面代码要正常运行，哪个选项正确（CD）

```java
interface Inter{
    void show();
}
public class Demo2 {
    public static void main(String[] args) {
        Inter i = new Inter() {
            @Override
            public void show() {
                System.out.println("你好！");
                System.out.println("我是Lambda！");
            }
        };
        method(i);
        // 想要达到上面相同的效果，选择哪个调用方式？
        method(??);
    }
    public void funA(){
      System.out.println("你好！");
      System.out.println("我是Lambda！");
    }
    public static void funB(){
      System.out.println("你好！");
      System.out.println("我是Lambda！");
    }
    public static void method(Inter i){
        i.show();
    }
}
-----------------------------------------
A. 
  method(Demo2::funA); 
B.
  Demo2 d = new Demo2();
  method(d::funB);
C.
  method(Demo2::funB); 
D. 
  Demo2 d = new Demo2();
  method(d::funA);
```

##### 6. 下面代码会导致编译错误的选项是（D）

```java
interface Inter{
    int[] show(int len);
}
public class Demo1 {
    public static void main(String[] args) {
        Inter i = new Inter() {
            @Override
            public int[] show(int len) {
                return new int[len];
            }
        };
        method(i);
        method(int[]::new); 
    }
   
    public static void method(Inter i,int len){ 
        i.show(len);
    }
}
-----------------------------------
A.
   Inter i = new Inter() {
      @Override
      public int[] show(int len) {
         return new int[len];
      }
   };
   method(i,3);
B.
   Inter i = a -> new int[a]; 
   method(i,0);
C.
   method(int[]::new,3);
D.
   method(new int[a],3);
```

##### 7. 下面针对数组排序算法，描述错误的是（C）

   A. 冒泡排序：元素两两比较，较大的元素会同相邻的小元素交换位置。

   B. 选择排序：每轮选择当前位置，开始找出后面的较小值与该位置交换。

   C. 冒泡排序是排序算法中效率最高的排序算法。

   D. 选择排序算法比冒泡排序算法，排序效率高。

##### 8. 【多选题】能够完全匹配字符串"back"和"back-end"的正则表达式包括（ ABCD ）

   A. `'\w{4}-\w{3}|\w{4}'`

   B. `'\w{4}|\w{4}-\w{3}'`

   C.`'\S+-\S+|\S+'`

   D. `'\w*-\w*|\w*'`

##### 9. 【多选题】能够完全匹配字符串"(010)-62661617"和"01062661617"的正则表达式包括（ABD） 

   A. `"\(?\d{3}\)?-?\d{8}"`

   B. `"[0-9()-]+"`

   C. `"[0-9(-)]*\d*"`

   D.`"[(]?\d*[)-]*\d*"`

------

#### 二、今日单词：

1. 数组工具类的单词：
2. 函数式接口的单词：
3. 方法引用的单词：
4. 冒泡排序方法的单词：
5. 二分查找方法的单词：

------

#### 三、简答题：

1. 如何使用Arrays工具类实现数组的降序排序？

   答：

2. 什么是面向函数式编程？同面向对象编程有何区别？

   答：

3. 请说明lambda表达式'()->{}'每个符号的含义？

   答：

4. 什么是方法引用？什么情况下可以使用方法引用？

   答：

5. 请说出冒泡排序和选择排序的排序原理？

   答：

6. 正则表达式区分语言吗？有何作用？

   答：


------

#### 四、排错题：

##### 排错题1：

```java
// 以下代码是否有问题？为什么？如何解决？
public class Demo1 { 
    public static void main(String[] args) {
        int[] arr = {5, 2, 3, 1, 4};
        Arrays.sort(arr, new Comparator<int>() {
            @Override
            public int compare(int o1, int o2) {
                return o2-o1;
            }
        });
        System.out.println(Arrays.toString(arr));
    }
}
```

##### 排错题2：

```java
// 下面代码是否能使用二分查找法，正确找到元素5的位置？如果不能，如何解决？
public class Demo2 {
  public static void main(String[] args) {
    int[] arr = {5, 2, 3, 1, 4};
    int index = Arrays.binarySearch(arr, 5);
    System.out.println(index);
  }
}
```

##### 排错题3：

```java
// 以下代码是否有问题？为什么？
public class Demo1 {
    public static void main(String[] args) {
        Inter i = new Inter() {
            @Override
            public void show() {
                System.out.println("匿名内部类 - show()");
            }
        };
        i.show();
    }
}
@FunctionalInterface
interface Inter{
    void show();
	String toString();
}
```

##### 排错题4：

![img](image\4.jpg)

上面lambda表达式的书写是否正确并说明原因？并给出正确的代码！

答：

------

#### 五、代码题：

##### 第一题：分析以下需求，并用代码实现

**训练目标**：掌握Java中数组排序，以及理解其在实际开发中的应用

**需求描述**：某学校要组织学生参加活动，要求学生按照身高从小打大进行排队。请在素材的sort项目中的Entry类中补全相关的代码，模拟排队效果！程序运行效果如下所示：

| ![image](image/image-20220126233746009.png) |
| ------------------------------------------------------------ |

**实现提示**：

1、使用冒泡排序比较两个学生的身高

2、再次使用选择排序比较两个学生的身高



------

##### 第二题：	分析以下需求，并用代码实现	

**训练目标**：掌握Java中lambda表达式的使用

**需求背景**：在注册网站用户的时候，大部分的网站都会先生成一个随机的验证码展示给用户，用于区分正常人和机器的操作。如下图所示：

| ![image-20220127204558689](image/image-20220127204558689.png) |
| ------------------------------------------------------------ |

**需求描述**：在素材的supplier项目的Entry类的指定位置补全代码(调用verificationCode方法)，实现一个生成四位随机验证码! 案例的执行效果如下所示：

| ![image-20220127205756468](image/image-20220127205756468.png) |
| ------------------------------------------------------------ |

**实现提示**：

1、Supplier<T>是一个生产类型接口，该接口的方法`T get();`是需要在方法内部生产一个T类型的对象返回。

2、verificationCode方法参数Supplier<String>说明如下所示：

| ![image-20220127210010293](image/image-20220127210010293.png) |
| ------------------------------------------------------------ |

3、生成验证码的字符可以从OPTIONS_CAHS中获取

4、调用verificationCode方法获取一个4位的随机验证码

------

##### 第三题：分析以下需求，并用代码实现

**训练目标**：掌握Java中lambda表达式的使用

**需求描述**：在素材的function项目中指定的位置补全代码，实现从userNames数组中随机获取一个元素返回(点名器)! 案例的执行效果如下所示：

![image-20220127161351575](image/image-20220127161351575.png)

**实现提示**：

1、Function<T,R>是一个转换类型接口，该接口的方法`R apply(T t);`是将传入对象T类型，转换成R类型并返回。

2、getUserName方法参数Function<String[] , String>说明如下所示：

![image-20220127161803471](image/image-20220127161803471.png)

3、使用lambda表达式完成getUserName方法调用

------

##### 第四题：【选做题】分析以下需求，并用代码实现

**训练目标**：掌握Java中二分查找代码的思想，以及在实际开发中的应用

**需求背景**：某公司的技术部门年会需要举行一个抽奖活动以感谢员工在这一年中的付出。奖品种类总共有4种：一等奖、二等级、三等奖、谢谢惠顾。为了节约成本需要控制每一种奖项的抽取概

率，常见的做法就是给每一个奖项分配一个int类型的权重值，权重越大抽取到的概率就越大。下图是具体的奖项权重值分配情况：

|  奖项  | 一等奖  | 二等奖  | 三等奖  | 谢谢惠顾 |
| :--: | :--: | :--: | :--: | :--: |
|  权重  |  1   |  2   |  3   |  4   |

**需求描述**：请在素材的lottery项目中Entry类中补全相关代码，完成按照权重随机抽奖程序的设计。多次运行程序(100次)，执行完毕以后在控制台的输出部分结果如下所示：

| ![image](image/image-20220126203429217.png) |
| ------------------------------------------------------------ |

**实现提示**：

1、所有奖项的总权重，以及每一个奖项的权重范围 , 如下所示：

|  奖项  |  一等奖  |  二等奖  |  三等奖  |  四等奖   |
| :--: | :---: | :---: | :---: | :----: |
| 权重范围 | [0,1) | [1,3) | [3,6) | [6,10) |

2、生成一个随机数作为权重数

3、利用二分查找根据权重数从奖项数组中获取对应的奖项

4、控制台输出奖项名称












