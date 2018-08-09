procedure Test is
   type My_Range is range 0 .. 20;
   type My_Enum is (A, B, C);
   type My_Array is array (My_Range, My_Enum) of Integer;
   x : My_Array;
begin
   x(12, C) := 14;
   x(12, A) := 14;
   x(12, B) := 14;
end Ex1;
