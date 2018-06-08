procedure Ex1 is
   type My_Range is range 0 .. 3;
   type My_Record is record
      x : Integer;
      y : Integer;
   end record;
   type My_Array is array (My_Range) of My_Record;
   x : My_Array;
   res : Integer;
begin
   x(2) := (12, 42);
   res := x(2).x;
   if x(3).x = 13 then
      res := x(2).y;
   end if;
end Ex1;
