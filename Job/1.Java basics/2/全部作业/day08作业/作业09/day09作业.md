##  day09作业

#### 一、选择题：

#####   1. 计算机中使用的ASCII码是对( C )的编码。   

A.英文字母

B.英文字母和数字

C.英文字符集

D.英文字符和中文字符

##### 2.标准ASCII码字符集共有( D )个字符编码。 

A.256

B.127

C.255

D.128

##### 3. Unicode(utf-8)和GBK编码描述正确的是（C）

A. utf-8只支持中文

B. GBK只支持中文

C. Unicode和GBK字符集支持英文也支持中文 

D. GBK不支持中文繁体字

##### 4. 字节输出流的抽象基类是(  A   ) 

A. OutputStream类  

B. InputStream类

C. Writer类

D. Reader类

##### 5. 如果在当前目录下不存在Hello.txt 文件，试图编译和运行下面代码会输出什么

```java
import java.io.*;
public class Mine {
    public static void main(String argv[]){
         Mine m=new Mine();
         System.out.println(m.amethod());
    }
    public int amethod() {
         try {
             FileInputStream dis= new FileInputStream("Hello.txt");
         }catch (FileNotFoundException fne) {
             System.out.println("No such file found");
             return -1;
         }catch(IOException ioe) {
         } finally{
             System.out.println("Doing finally");
         }
         return 0;
    }
}
```

A. No such file found
B. No such file found ,-1
C. No such file found, Doing finally, -1	
D. 0

##### 6. 下面程序的运行结果是( D   ) 

```java
public static void main(String argv[]){ 
   FileOutputStream fos = newFileOutputStream(“c:\demo.txt”);
   fos.write("abc");   
   fos.close();
}
```

A. 在C盘创建文件demo.txt,但文件是空的

B. 在C盘创建文件demo.txt,并写入数据abc

C. 将C盘已有的文件demo.txt中追加写入abc

D. 编译失败

 ##### 7. 当方法遇到异常(编译时异常)又不知如何处理时，下列哪种说法是正确的(  C  )

A. 捕获异常（try{…}catch{…}）

B. 抛出异常(throw 异常对象)

C. 声明异常（throws 异常类型）

D. 嵌套异常

##### 8. 【多选题】下面关于try-catch-finally结构的说法错误的是(AB) 

A.—个try块后面必须至少带一个catch块或finally块

B.—个try块后面可以带一个或者多个catch块或者finally块

C.try-catch-finally结构中finally块可以没有

D.try-catch-finally结构中如有finally块，则finally块一定被执行 


##### 9. 下面的程序创建了一个文件输出流对象，用来向文件test.txt中输出数据，假设程序当前目录下不存在文件test.txt，编译下面的程序Test.java后，将该程序运行3次，则文件test.txt 的内容是(D  )

```java
import java.io.*;
public class Test {
	public static void main(String args[]) {
		try {
			String s="ABCDE ";
			byte b[]=s.getBytes();
			FileOutputStream file= new FileOutputStream("test.txt",true); //看这里
			file.write(b);
			file.close();
		}
		catch(IOException e) {
			System.out.println(e.toString());
		}
	}
}
```

A. ABCABC

B. ABCDE

C. Test

D. ABCDE ABCDE ABCDE

##### 10.  下面说法不正确的是 (    ) 

A. final是个关键字，用来修饰成员变量(常量)、成员方法(不能被重写)、类(不能被继承)

B. finally是异常处理的一部分，它里面的代码正常情况下永远被执行

C. finalize是Object类中定义的，用于垃圾回收，由垃圾回收器调用。

D. final、finally、finalize三个关键字都可以在类的内部

##### 11. 【多选题】如果需要从文件中读取数据，则可以在程序中创建哪一个类的对象（AD）

A.FileInputStream

B.FileOutputStream   

C. DataOutputStream   

D. BufferedInputStream

##### 12. 文件输出流的构造方法是public FileOutputStream(String name,Boolean append) throwsFileNotFoundException，当参数append的值为true时，表示( B )

A.创建一个新文件

B. 在原文件的尾部添加数据

C. 覆盖原文件的内容

D. 在原文件的指定位置添加数据

------

#### 二、今日方法：

1. String类中能完成编码、解码的方法名：

   ```java
   答：
   ```

2. IO流中的字节输入流所涉及到的类名，以及方法名：

   ```java
   答：
   ```

3. IO流中的字节输出流所涉及到的类名，以及方法名：

   ```java
   答：
   ```

------

#### 三、简答题：

1. 什么是字符集，有哪些常见的字符集？

   ```java
   答：
   ```

2. UTF-8和GBK有什么区别？

   ```java
   答：
   ```

3. IO流是用来做什么的？IO流如何分类的？

   ```java
   答：
   ```

4. IO流中的字节流可以操作什么类型的文件？有哪些注意事项？

   ```java
   答：
   ```

5. try...catch...finaly中的finaly用来干嘛的？有何特点？能不能直接写try...finaly？

   ```java
   答：
   ```


------

#### 四、排错题：

##### 排错题1：

```java
// 以下代码是否能正常运行？结果是什么？
public class Demo3 {
       public static void main(String[] args) {
        int a = show();
        System.out.println(a);
    }

    public static int show() {
        try {
            System.out.println(1 / 0);
            System.out.println(111);
            return 1;
        } catch (Exception e) {
            System.out.println(222);
            return 2;
        } finally {
            System.out.println(333);
            return 3;
        }
    }
}
```

```tex
答：
```

##### 排错题2：

```java
// 以下代码运行后，控制台输出结果是什么？ 如何解决？
public class Demo3 {
    public static void main(String[] args) throws UnsupportedEncodingException {
        String s = "黑马IT全国第一";
        // 编码
        byte[] bytes = s.getBytes();
        // 解码
        String s2 = new String(bytes,"gbk");
        System.out.println(s2);
    }
}
```

```java
答：
```

##### 排错题3：

```java
// 以下代码，运行后会有问题吗？为什么？文件中会有什么数据？
// 文件abc.txt已有数据：中国你好！
public class Demo4 {
    public static void main(String[] args) throws IOException {
        FileInputStream fis = new FileInputStream("d:\\abc.txt");
        FileOutputStream fos = new FileOutputStream("d:\\abc.txt",true);
        int len;
        while((len = fis.read())!= -1){
            fos.write("你好呀~".getBytes());
        }
        fis.close();
        fos.close();
    }
}
```

```java
答：
```

------

#### 五、代码题：

##### 第一题：分析以下需求，并用代码实现

**训练目标**：

​	掌握java中字节输入流和字节输出流的基本使用，以及理解其在实际开发中的应用

**需求背景**：

​	为了保证磁盘文件数据的安全性，就需要对原始文件进行加密存储，再使用的时候再对其进行解密处理。加密解密原理：对原始文件中的每一个字节数据进行更改，然后将更改以后的数据存储到新的文件中。

**需求描述**: 

​	编写一个加密解密程序。

**实现提示**：

​	1、通过字节输入流读取原始文件中的每一个字节数据然后对其进行处理

​	2、通过字节输出流将处理以后的字节数据写入到另外一个文件中

​	3、对字节数据的处理方式可以使用异或运算符异或一个数

​	4、为了提高加解密的效率可以考虑使用高效的字节输入和输出流



------

##### 第二题：	分析以下需求，并用代码实现	

**训练目标**：

​	掌握java中字节输入流和字节输出流的基本使用

**需求背景**：

​	将D盘中某文件夹中的所有文件，复制到E盘中某文件夹中，要求E盘复制的文件夹，必须同D盘中被赋值的文件夹名称一致，文件夹中的内容也必须一致。

**需求描述**: 

​	模仿windows中的，Ctrl + C和Ctrl + V功能，编写一个文件复制程序。

****实现提示****：

	1. 必须使用递归，才能拿到D盘中的每一个文件和文件夹，进行复制
	2. 通过File对象可以获取文件夹和文件的名称，这样才能保证文件名一致
	3. 使用IO流中的字节流对文件进行复制

------

##### 第三题：【选做题】分析以下需求，并用代码实现

**训练目标**：

​	掌握java中字节输入流和字节输出流的基本使用。

**需求描述**：

​	文件复制的同时，在控制台显示文件复制的进度，进度以百分比表示。

**实现提示**：

```java
DecimalFormat df = new DecimalFormat("##%");
String format = df.format(3.0 / 10);
System.out.println(format);  // 30%
```

![](./image/1.jpg)






