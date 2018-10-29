procedure Test is
   type My_Array is array (Integer) of Integer;
   x : My_Array;
begin
   x(0) := 0;
   while x(0) < 100000 loop
      x(0) := x(0) + 1;
   end loop;
end Ex1;
