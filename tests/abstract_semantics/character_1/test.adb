procedure Ex1 is
   subtype My_Index is Integer range 1 .. 5;
   type My_String is array (My_Index) of Character;
   X : My_String := "hello";
begin
   X(1) := ''';
end Ex1;
