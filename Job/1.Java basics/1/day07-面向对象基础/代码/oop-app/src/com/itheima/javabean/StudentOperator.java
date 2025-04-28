<<<<<<< HEAD
package com.itheima.javabean;

public class StudentOperator {
    private Student student;
    public StudentOperator(Student student){
        this.student = student;
    }

    public void printPass(){
        if(student.getScore() >= 60){
            System.out.println(student.getName() + "学生成绩及格");
        }else {
            System.out.println(student.getName() + "学生成绩不及格");
        }
    }
}
=======
package com.itheima.javabean;

public class StudentOperator {
    private Student student;
    public StudentOperator(Student student){
        this.student = student;
    }

    public void printPass(){
        if(student.getScore() >= 60){
            System.out.println(student.getName() + "学生成绩及格");
        }else {
            System.out.println(student.getName() + "学生成绩不及格");
        }
    }
}
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
