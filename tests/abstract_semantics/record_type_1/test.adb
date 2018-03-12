procedure Ex1 is
   type Point is record
      x : Integer;
      y : Integer;
      z : Integer;
   end record;
   z : Integer;
   p : Point := (z, 3, 12);
begin
   if p = (3, z, 12) then
      p := (x | z => 31, y => 41);
   else
      p := (x => 4, others => 2);
   end if;
end Ex1;
