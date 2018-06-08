procedure Ex1 is
   type My_Range is range 0 .. 20;
   type My_Array is array (My_Range) of Integer;
   x : My_Array;
   res : Integer;
   i : My_Range;
begin
   i := My_Range'First;
   while i /= My_Range'Last loop
      x(i) := 12;
      i := i + 1;
   end loop;

   res := x(10);
end Ex1;
