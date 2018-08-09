procedure Test is
   type Integer_Access is access all Integer;
   x : Integer_Access;
   y : Integer := 0;
begin
   y := x.all;

   if x = null then
      y := 2;
   end if;
end Ex1;
