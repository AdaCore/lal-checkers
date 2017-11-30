procedure Ex1 is
   type Integer_Access is access all Integer;
   x : Integer_Access;
   y : Integer := 0;
begin
   if x /= null and then x.all = y then
      y := 42;
   end if;
end Ex1;
