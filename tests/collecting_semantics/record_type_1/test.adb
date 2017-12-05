procedure Ex1 is
   type Point is record
      x : Integer;
      y : Integer;
   end record;
   z : Integer;
   p : Point := (z, 3);
begin
   if p = (3, z) then
      p := (31, 41);
   end if;
end Ex1;
