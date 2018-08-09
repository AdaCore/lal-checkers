procedure Test is
   type My_Range is range 0 .. 3;
   type My_Array is array (My_Range) of Integer;
   x : My_Array;
   res : Integer;
begin
   x(2) := 42;
   res := x(2);
   if x(3) = 13 then
      res := x(3);
   end if;
end Ex1;
